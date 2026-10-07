"""수집 → 정제 → 조각 임베딩 → 문장 인덱스를 한 번에 실행한다.

각 단계는 이미 증분 처리(내용이 바뀐 파일·벡터만 갱신)를 하므로, 바뀐 게 없으면 빨리 지나간다.
임베딩 모델(bge-m3)은 한 번만 불러와 조각 임베딩과 문장 인덱스가 함께 쓴다.

수집은 기본적으로 캐시(raw HTML)만 쓴다. 외부 요청이 필요한 새 수집은 --crawl을 붙여야 한다.

사용 예:
    python pipeline.py                          # 모든 책: 수집(캐시) → 정제 → 조각 임베딩 → 문장 인덱스
    python pipeline.py --book-id 110            # 특정 책만
    python pipeline.py --from embed             # 조각 임베딩부터
    python pipeline.py --to refine              # 정제까지
    python pipeline.py --rebuild --from embed   # 벡터 저장소를 archive로 옮기고 새로 생성
    python pipeline.py --crawl --book-id 2      # 새 책 수집 (위키독스에 외부 요청)
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

import evidence
import text_embed
import text_ingestion
import text_refine
from text_ingestion import DATA_ROOT, Paths, move_to_archive

STAGES = ("ingest", "refine", "embed", "evidence")
STAGE_LABELS = {
    "ingest": "수집",
    "refine": "정제",
    "embed": "조각 임베딩",
    "evidence": "문장 인덱스",
}
BOOK_DIR = re.compile(r"book(\d+)")

logger = logging.getLogger("pipeline")


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
        self.chunk_collection = None
        self.sentence_collection = None

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
        text_refine.refine_book(
            self.book_dir(book_id),
            text_refine.DEFAULT_MIN_CHARS,
            text_refine.DEFAULT_MAX_CHARS,
        )

    def embed(self, book_id: int) -> None:
        text_embed.index_book(self.book_dir(book_id), self.chunk_collection, self.model)

    def evidence(self, book_id: int) -> None:
        evidence.index_book(
            self.book_dir(book_id), self.sentence_collection, self.model
        )

    def run(self, stages: list[str]) -> None:
        if self.rebuild:
            # 저장소를 지우지 않고 archive로 옮긴다
            if "embed" in stages:
                move_to_archive(text_embed.VECTORSTORE_DIR)
            if "evidence" in stages:
                move_to_archive(evidence.SENTENCE_STORE_DIR)
        if "embed" in stages:
            self.chunk_collection = text_embed.get_collection()
        if "evidence" in stages:
            self.sentence_collection = evidence.get_sentence_collection()

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
            print(f"  {STAGE_LABELS[stage]:<7} book{book_id:<5} {sec:7.1f}초")
        print(f"  합계 {sum(s for _, _, s in elapsed):.1f}초")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="수집 → 정제 → 조각 임베딩 → 문장 인덱스 원클릭 실행"
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
        help="범위 안의 벡터 저장소(조각·문장)를 archive로 옮기고 새로 생성",
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
