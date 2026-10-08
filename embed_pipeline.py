"""수집 → 정제 → 문단 임베딩을 한 번에 실행한다.

각 단계는 이미 증분 처리(내용이 바뀐 파일·벡터만 갱신)를 하므로, 바뀐 게 없으면 빨리 지나간다.
문단 임베딩 단계는 chunks.jsonl(페이지 하나 = 한 줄)에서 A+ 문단을 만들어 bge-m3로 임베딩한다.
문단은 파일로 저장하지 않는다 (조각·소제목·문장 단위의 옛 임베딩은 폐기했다).
--rebuild는 정제를 건너뛰지 않고 다시 하고, 벡터 저장소를 archive로 옮겨 새로 만들되
임베딩 텍스트가 같은 문단의 벡터는 옛 저장소에서 가져와 재사용한다.

수집은 기본적으로 캐시(raw HTML)만 쓴다. 외부 요청이 필요한 새 수집은 --crawl을 붙여야 한다.

사용 예:
    python embed_pipeline.py                          # 모든 책: 수집(캐시) → 정제 → 문단 임베딩
    python embed_pipeline.py --book-id 110            # 특정 책만
    python embed_pipeline.py --from embed             # 문단 임베딩부터
    python embed_pipeline.py --to refine              # 정제까지
    python embed_pipeline.py --rebuild --from refine  # 건너뛰기 없이 정제부터 다시 (저장소는 archive로 옮기고 새로 생성)
    python embed_pipeline.py --rebuild --from embed   # 벡터 저장소를 archive로 옮기고 새로 생성
    python embed_pipeline.py --crawl --book-id 2      # 새 책 수집 (위키독스에 외부 요청)
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

import text_embed
import text_ingestion
import text_paragraphs
import text_refine
from text_ingestion import DATA_ROOT, Paths, move_to_archive

STAGES = ("ingest", "refine", "embed")
STAGE_LABELS = {
    "ingest": "수집",
    "refine": "정제",
    "embed": "문단 임베딩",
}
BOOK_DIR = re.compile(r"book(\d+)")

logger = logging.getLogger("embed_pipeline")


def existing_book_ids() -> list[int]:
    ids = [
        int(m.group(1))
        for p in DATA_ROOT.iterdir()
        if (m := BOOK_DIR.fullmatch(p.name))
    ]
    return sorted(ids)


class Pipeline:
    def __init__(self, book_ids: list[int], crawl: bool, rebuild: bool):
        self.book_ids = book_ids
        self.crawl = crawl
        self.rebuild = rebuild
        self._model = None
        self.collection = None
        self.reuse: dict[str, list[float]] | None = None  # --rebuild: 재사용할 벡터

    @property
    def model(self):
        if self._model is None:
            self._model = text_embed.load_model()
        return self._model

    def book_dir(self, book_id: int) -> Path:
        return Paths.for_book(book_id).book_dir

    def ingest(self, book_id: int) -> None:
        if self.crawl:
            logger.warning("book%d: 위키독스에 외부 요청을 보냅니다 (--crawl)", book_id)
        text_ingestion.run(
            book_id=book_id,
            delay=float(
                os.getenv("WIKIDOCS_DELAY_SEC", text_ingestion.DEFAULT_DELAY_SEC)
            ),
            user_agent=os.getenv(
                "WIKIDOCS_USER_AGENT", text_ingestion.DEFAULT_USER_AGENT
            ),
            offline=not self.crawl,
        )

    def refine(self, book_id: int) -> None:
        # 입력(pages.jsonl, 설정)이 그대로면 건너뛴다. --rebuild는 건너뛰지 않고 다시 정제한다
        text_refine.refine_book(self.book_dir(book_id), force=self.rebuild)

    def embed(self, book_id: int) -> None:
        # 모델은 새로 임베딩할 문단이 있을 때만 불러온다
        text_paragraphs.index_book(
            self.book_dir(book_id), self.collection, lambda: self.model, self.reuse
        )

    def run(self, stages: list[str]) -> None:
        if self.rebuild and "embed" in stages:
            # 저장소를 지우지 않고 archive로 옮기고, 거기서 읽은 벡터를 다시 쓴다
            archived = move_to_archive(text_paragraphs.VECTORSTORE_DIR)
            self.reuse = text_paragraphs.read_embeddings(archived) if archived else None
        if "embed" in stages:
            self.collection = text_paragraphs.get_collection()

        elapsed: list[tuple[str, int, float]] = []
        for stage in stages:
            for book_id in self.book_ids:
                label = f"[{STAGE_LABELS[stage]}] book{book_id}"
                logger.info("%s 시작", label)
                start = time.perf_counter()
                try:
                    getattr(self, stage)(book_id)
                except Exception:
                    # 앞 단계가 실패한 채로 뒷 단계가 옛 데이터로 돌지 않게 멈춘다
                    logger.exception("%s 실패 — 파이프라인을 멈춥니다", label)
                    self.summary(elapsed)
                    raise
                elapsed.append((stage, book_id, time.perf_counter() - start))
        self.summary(elapsed)

    @staticmethod
    def summary(elapsed: list[tuple[str, int, float]]) -> None:
        if not elapsed:
            return
        print("\n=== 파이프라인 요약 ===")
        for stage, book_id, sec in elapsed:
            print(f"  {STAGE_LABELS[stage]:<9} book{book_id:<5} {sec:7.1f}초")
        print(f"  합계 {sum(s for _, _, s in elapsed):.1f}초")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="수집 → 정제 → 문단 임베딩 원클릭 실행"
    )
    parser.add_argument(
        "--book-id",
        type=int,
        nargs="+",
        default=None,
        help="생략하면 .data의 모든 bookN",
    )
    parser.add_argument(
        "--from", dest="start", choices=STAGES, default=STAGES[0], help="시작 단계"
    )
    parser.add_argument(
        "--to", dest="end", choices=STAGES, default=STAGES[-1], help="끝 단계"
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="수집 단계에서 캐시에 없는 페이지를 위키독스에서 받는다 (외부 요청)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="범위 안의 단계를 건너뛰지 않고 처음부터 다시 만든다 "
        "(정제는 다시 파싱, 임베딩은 벡터 저장소를 archive로 옮기고 새로 생성)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    first, last = STAGES.index(args.start), STAGES.index(args.end)
    if first > last:
        parser.error(f"--from {args.start}가 --to {args.end}보다 뒤입니다")
    book_ids = args.book_id or existing_book_ids()
    if not book_ids:
        parser.error(
            ".data에 bookN 폴더가 없습니다. --crawl --book-id N으로 수집하세요"
        )
    stages = list(STAGES[first : last + 1])
    logger.info("책 %s, 단계 %s", book_ids, " → ".join(STAGE_LABELS[s] for s in stages))
    Pipeline(book_ids, args.crawl, args.rebuild).run(stages)


if __name__ == "__main__":
    main()
