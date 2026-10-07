"""문단 인덱스: 조각(chunks.jsonl)에서 A+ 문단을 만들고, bge-m3로 임베딩해 Chroma에 저장·검색한다.

설계: .idea_folder/정확도_2차_근거문장_인덱스_설계.md (0장). 문장·조각 단위 인덱스는 폐기했다.

A+ 문단 = 설명 블록(paragraph, note, key_point, caption, concept_box, 연속한 list)과 바로 뒤의
코드·실행 예시를 한 묶음(B)으로 보고, B를 같은 소제목 안에서 이웃과 합쳐 목표 길이(200자)까지
키운 것이다 (상한 600자, 100자 미만의 짧은 꼬리는 앞 단위에 붙인다).
소제목이 없는 구간은 페이지 제목으로 대체한다. 한 페이지에서 그렇게 대체한 단위가 여럿이면
표시용 라벨을 "제목 - 1", "제목 - 2"로 구분한다. `[이미지: …]` 표시는 텍스트에서 빼고
`has_image`, `figure_ref`("그림 N" 참조) 플래그로만 남긴다.

저장:
    .data/bookN/paragraphs.jsonl            문단 원본 (텍스트·소제목·URL·블록 범위·플래그)
    .data/vectorstore_paragraphs/           Chroma. 문단마다 벡터 하나 (embed_hash로 증분 갱신)

사용 예:
    python paragraphs.py                                  # 모든 책 증분 임베딩
    python paragraphs.py --book-id 1                      # 특정 책만
    python paragraphs.py --rebuild                        # 저장소를 archive로 옮기고 새로 생성
    python paragraphs.py --query "변수" --book-id 110     # 문단 검색
"""

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from text_embed import (
    CODE_BLOCKS,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    block_to_text,
    embed,
    load_chunk_index,
    load_chunks,
    load_model,
    render_block,
    render_blocks,
)
from text_ingestion import DATA_ROOT, Paths, move_to_archive

VECTORSTORE_DIR = DATA_ROOT / "vectorstore_paragraphs"
COLLECTION_NAME = "paragraphs_bge-m3"
PARAGRAPHS_FILE = "paragraphs.jsonl"
UPSERT_BATCH_SIZE = 64

EXPLAIN_BLOCKS = ("paragraph", "note", "key_point", "caption", "concept_box")
MIN_UNIT_CHARS = 30
PLUS_TARGET_CHARS = 200  # 이 길이가 될 때까지 이웃 단위를 합친다
PLUS_MAX_CHARS = 600  # 합친 길이가 이를 넘으면 합치지 않는다
PLUS_TAIL_CHARS = 100  # 구간 끝에 남은 이보다 짧은 꼬리는 앞 단위에 붙인다
CORE_MARK = "<<핵심 문단>>"  # 페이지 맥락 안에서 핵심 문단이 있던 자리
PAGE_CONTEXT_CHARS = 4000  # 페이지 맥락 최대 글자 수
# "[이미지: 02_3_list.png]" 표시는 LLM에 정보가 없고 글자만 차지해서 뺀다 (뒤따르는 공백까지)
IMAGE_MARK = re.compile(r"\[이미지: [^\]]*\]\s*")
FIGURE_REF = re.compile(r"그림\s?\d+(\.\d+)*")  # 내용이 그림에 있다는 신호

FETCH_LIMIT = 300  # 검색할 때 벡터에서 가져오는 후보 수 (관문을 거치고 나면 줄어든다)
GATE_STOPWORDS = {"자료형", "파이썬", "파이썬의", "값을", "저장하는", "공간", "기본"}
GATE_ALIASES = {
    "불리언": ["boolean", "bool"],
    "불": ["bool", "불 자료형"],
    "숫자형": ["숫자"],
}

logger = logging.getLogger("paragraphs")


# ---------------------------------------------------------------- 문단 만들기


def anchor_url(chunk: dict, anchor: str | None) -> str:
    base = chunk["source_url"].split("#")[0]
    return f"{base}#{anchor}" if anchor else chunk["source_url"]


def sections(blocks: list[dict]) -> list[tuple[str, str | None]]:
    """블록마다 (소속 소제목, 앵커). h2/h3와 subheading이 소제목을 바꾼다."""
    current: tuple[str, str | None] = ("", None)
    result = []
    for block in blocks:
        if block["type"] == "heading" and block["level"] in (2, 3):
            current = (block["text"], block.get("anchor"))
        elif block["type"] == "subheading":
            current = (block["text"], current[1])
        result.append(current)
    return result


def explain_ranges(blocks: list[dict]) -> list[tuple[int, int]]:
    """설명 블록의 범위. 연속한 list 항목은 한 단위로 묶는다."""
    ranges: list[tuple[int, int]] = []
    n = 0
    while n < len(blocks):
        kind = blocks[n]["type"]
        if kind == "list":
            end = n
            while end < len(blocks) and blocks[end]["type"] == "list":
                end += 1
            ranges.append((n, end))
            n = end
            continue
        if kind in EXPLAIN_BLOCKS:
            ranges.append((n, n + 1))
        n += 1
    return ranges


def with_code(
    blocks: list[dict], ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """설명 범위마다 바로 뒤에 이어지는 코드·실행 예시 블록까지 넓힌다."""
    result = []
    for start, end in ranges:
        while end < len(blocks) and blocks[end]["type"] in CODE_BLOCKS:
            end += 1
        result.append((start, end))
    return result


def merge_short(
    blocks: list[dict], secs: list, ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """범위들을 같은 소제목 안에서 이웃과 합쳐 목표 길이(PLUS_TARGET_CHARS)까지 키운다.

    합친 길이가 PLUS_MAX_CHARS를 넘으면 합치지 않는다. 사이에 다른 블록(제목, 표 등)이
    끼면 끊는다. 구간 끝에 남은 짧은 꼬리(PLUS_TAIL_CHARS 미만)는 앞 단위에 붙인다(상한 안에서).
    """

    def length(start: int, end: int) -> int:
        return len(render_blocks(blocks[start:end]))

    def mergeable(prev: tuple[int, int], nxt: tuple[int, int]) -> bool:
        return prev[1] == nxt[0] and secs[prev[0]] == secs[nxt[0]]

    merged: list[tuple[int, int]] = []
    for cur in ranges:
        if merged and mergeable(merged[-1], cur):
            prev = merged[-1]
            if (
                length(*prev) < PLUS_TARGET_CHARS
                and length(prev[0], cur[1]) <= PLUS_MAX_CHARS
            ):
                merged[-1] = (prev[0], cur[1])
                continue
        merged.append(cur)
    result: list[tuple[int, int]] = []
    for cur in merged:  # 짧은 꼬리는 앞 단위에 붙인다
        if (
            result
            and length(*cur) < PLUS_TAIL_CHARS
            and mergeable(result[-1], cur)
            and length(result[-1][0], cur[1]) <= PLUS_MAX_CHARS
        ):
            result[-1] = (result[-1][0], cur[1])
        else:
            result.append(cur)
    return result


def embed_text(title: str, section: str, chosen: list[dict]) -> str:
    """임베딩 텍스트: [페이지 제목] 소제목 + 본문. 소제목이 없으면 페이지 제목만 머리말로 쓴다."""
    head = f"[{title}] {section}" if section and section != title else f"[{title}]"
    body = "\n".join(t for b in chosen if (t := block_to_text(b)))
    return f"{head}\n{body}".strip()


def build_units(chunk: dict) -> list[dict]:
    """조각의 A+ 문단 목록. 각 문단은 블록 범위(block_start, block_end)를 갖는다."""
    blocks = chunk["blocks"]
    secs = sections(blocks)
    ranges = merge_short(blocks, secs, with_code(blocks, explain_ranges(blocks)))
    # 조각 path의 끝은 조각의 첫 소제목이라 페이지 제목이 아니다. 페이지 제목을 쓴다
    title = chunk.get("page_title") or (chunk.get("path") or [""])[-1]

    units: list[dict] = []
    for start, end in ranges:
        chosen = blocks[start:end]
        raw = render_blocks(chosen)
        text = IMAGE_MARK.sub("", raw).strip()
        if len(text) < MIN_UNIT_CHARS:
            continue
        section, anchor = secs[start]
        body = embed_text(title, section, chosen)
        units.append(
            {
                "unit_id": f"{chunk['chunk_id']}#g{len(units) + 1:03d}",
                "chunk_id": chunk["chunk_id"],
                "page_id": chunk.get("page_id"),
                "book_id": chunk.get("book_id"),
                "path": chunk.get("path", []),
                "section": section or title,
                "fallback": not section,
                "url": anchor_url(chunk, anchor),
                "text": text,
                "has_image": bool(IMAGE_MARK.search(raw)),
                "figure_ref": bool(FIGURE_REF.search(text)),
                "block_start": start,
                "block_end": end,
                "embed_text": body,
                "embed_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            }
        )
    # 페이지 제목으로 대체한 단위가 여럿이면 "제목 - 1", "제목 - 2"로 구분한다 (표시용)
    fallbacks = [u for u in units if u["fallback"]]
    if len(fallbacks) > 1:
        for n, unit in enumerate(fallbacks, start=1):
            unit["section"] = f"{title} - {n}"
    return units


# ---------------------------------------------------------------- 저장소


def get_collection():
    # 순수 Python protobuf 구현으로 우회한다 (conda-forge chromadb의 opentelemetry-proto 호환 문제)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),  # 외부 전송 차단
    )
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embed_model": MODEL_NAME},
    )


def metadata(row: dict, source_dir: str) -> dict:
    return {
        "unit_id": row["unit_id"],
        "chunk_id": row["chunk_id"],
        "page_id": row["page_id"] if row["page_id"] is not None else -1,
        "book_id": row["book_id"] if row["book_id"] is not None else -1,
        "source_dir": source_dir,
        "path": " > ".join(row["path"]),
        "section": row["section"],
        "embed_model": MODEL_NAME,
        "embed_hash": row["embed_hash"],
    }


def write_rows(path: Path, rows: list[dict]) -> bool:
    """내용이 바뀐 때만 쓴다. 바뀌었으면 True."""
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def index_book(book_dir: Path, collection, model) -> None:
    """chunks.jsonl → paragraphs.jsonl → 벡터 저장소. 임베딩 텍스트가 바뀐 문단만 다시 임베딩한다."""
    rows = [u for c in load_chunks(book_dir / "chunks.jsonl") for u in build_units(c)]
    changed = write_rows(book_dir / PARAGRAPHS_FILE, rows)

    got = collection.get(where={"source_dir": book_dir.name}, include=["metadatas"])
    existing = {
        i: m.get("embed_hash", "") for i, m in zip(got["ids"], got["metadatas"])
    }
    todo = [r for r in rows if existing.get(r["unit_id"]) != r["embed_hash"]]
    stale = set(existing) - {r["unit_id"] for r in rows}
    if stale:
        logger.warning(
            "%s: 더 이상 없는 벡터 %d개가 남아 있음. --rebuild로 정리하세요",
            book_dir.name,
            len(stale),
        )
    logger.info(
        "%s: 문단 %d개 (%s), 새로 임베딩 %d개",
        book_dir.name,
        len(rows),
        "paragraphs.jsonl 갱신" if changed else "paragraphs.jsonl 그대로",
        len(todo),
    )
    if not todo:
        return
    too_long = [
        r["unit_id"]
        for r in todo
        if len(model.tokenizer(r["embed_text"])["input_ids"]) > MAX_SEQ_LENGTH
    ]
    if too_long:
        logger.warning("%d토큰 초과로 잘리는 문단: %s", MAX_SEQ_LENGTH, too_long)
    for start in range(0, len(todo), UPSERT_BATCH_SIZE):
        batch = todo[start : start + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[r["unit_id"] for r in batch],
            embeddings=embed(model, [r["embed_text"] for r in batch]),
            documents=[r["embed_text"] for r in batch],
            metadatas=[metadata(r, book_dir.name) for r in batch],
        )
        logger.info("upsert %d/%d", start + len(batch), len(todo))


# ---------------------------------------------------------------- 검색


def concept_tokens(concept: str) -> list[str]:
    """페이지 관문용 핵심어. 'if문' → 'if', '튜플 자료형' → '튜플'."""
    words = [w for w in re.split(r"[\s,]+", concept) if w]
    tokens = []
    for word in words:
        if re.fullmatch(r"[A-Za-z]+문", word):
            word = word[:-1]
        if len(word) >= 2 and word not in GATE_STOPWORDS:
            tokens.append(word.lower())
    for word in words:
        tokens += GATE_ALIASES.get(word, [])
    return tokens or [concept.lower()]


def page_gate(concept: str, hits: list[dict]) -> list[dict]:
    """개념 핵심어가 페이지 경로에 있는 문단만 남긴다. 하나도 없으면 그대로 둔다."""
    tokens = concept_tokens(concept)
    kept = [h for h in hits if any(t in " > ".join(h["path"]).lower() for t in tokens)]
    return kept or hits


class ParagraphIndex:
    """개념·질문에 맞는 문단을 점수순으로 찾는다. 문단 원본은 책별 paragraphs.jsonl에서 읽는다."""

    def __init__(self, model=None, collection=None, chunk_index=None):
        self.model = model or load_model()
        self.collection = collection or get_collection()
        self._chunk_index = chunk_index
        self._rows: dict[int, dict[str, dict]] = {}

    @property
    def chunk_index(self) -> dict[str, dict]:
        if self._chunk_index is None:
            self._chunk_index = load_chunk_index()
        return self._chunk_index

    def rows(self, book_id: int) -> dict[str, dict]:
        if book_id not in self._rows:
            path = Paths.for_book(book_id).book_dir / PARAGRAPHS_FILE
            if not path.exists():
                raise FileNotFoundError(f"{path} 없음 (먼저 python paragraphs.py 실행)")
            with path.open(encoding="utf-8") as f:
                self._rows[book_id] = {r["unit_id"]: r for r in map(json.loads, f) if r}
        return self._rows[book_id]

    def find(
        self,
        concept: str,
        book_id: int | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> list[dict]:
        """문단을 점수(코사인 유사도)순으로. 페이지 관문을 거친다. exclude는 이미 쓴 unit_id."""
        query = embed(self.model, [concept])
        result = self.collection.query(
            query_embeddings=query,
            n_results=FETCH_LIMIT,
            where={"book_id": book_id} if book_id is not None else None,
            include=["metadatas", "distances"],
        )
        found = []
        for unit_id, meta, dist in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0]
        ):
            if unit_id in exclude:
                continue
            row = self.rows(meta["book_id"]).get(unit_id)
            if row is None:  # 규칙이 바뀌어 남은 옛 벡터
                continue
            if row["embed_hash"] != meta.get("embed_hash"):
                continue
            found.append({**row, "score": round(1 - dist, 4), "distance": dist})
        return sorted(page_gate(concept, found), key=lambda u: -u["score"])

    def page_context(self, unit: dict, limit: int = PAGE_CONTEXT_CHARS) -> str:
        """핵심 문단이 속한 페이지(조각). 핵심 문단은 CORE_MARK 한 줄로 바꿔 중복해서 넣지 않는다.

        길이가 limit를 넘으면 핵심 문단에서 앞뒤로 블록을 번갈아 넓혀 가며 limit 안에서 자른다.
        """
        blocks = self.chunk_index[unit["chunk_id"]]["blocks"]
        rendered = [IMAGE_MARK.sub("", render_block(b)).strip() for b in blocks]
        start, end = unit["block_start"], unit["block_end"]
        budget = limit - len(CORE_MARK)
        lo, hi, used = start, end, 0
        while True:
            progressed = False
            if lo > 0 and used + len(rendered[lo - 1]) + 1 <= budget:
                lo -= 1
                used += len(rendered[lo]) + 1
                progressed = True
            if hi < len(rendered) and used + len(rendered[hi]) + 1 <= budget:
                used += len(rendered[hi]) + 1
                hi += 1
                progressed = True
            if not progressed:
                break
        parts = [r for r in rendered[lo:start] if r]
        parts.append(CORE_MARK)
        parts += [r for r in rendered[end:hi] if r]
        if lo > 0:
            parts.insert(0, "...(앞부분 생략)")
        if hi < len(rendered):
            parts.append("...(뒷부분 생략)")
        return "\n".join(parts)

    def vectors(self, unit_ids: list[str]) -> np.ndarray:
        """저장된 문단 벡터 (unit_ids 순서)."""
        got = self.collection.get(ids=unit_ids, include=["embeddings"])
        by_id = dict(zip(got["ids"], got["embeddings"]))
        return np.asarray([by_id[i] for i in unit_ids], dtype=np.float32)


# ---------------------------------------------------------------- 실행


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="문단 인덱스 (bge-m3 + Chroma)")
    parser.add_argument("--book-id", type=int, default=None, help="생략하면 모든 책")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="저장소를 archive로 옮기고 새로 생성",
    )
    parser.add_argument("--query", help="인덱스 갱신 대신 문단 검색")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.query:
        found = ParagraphIndex().find(args.query, args.book_id)
        print(f"문단 {len(found)}개")
        for n, unit in enumerate(found[: args.top], start=1):
            text = " ".join(unit["text"].split())[:90]
            print(f"{n:>3}. {unit['score']:.3f} [{unit['section']}] {text}")
        return

    if args.rebuild:
        move_to_archive(VECTORSTORE_DIR)
    if args.book_id is not None:
        book_dirs = [Paths.for_book(args.book_id).book_dir]
    else:
        book_dirs = sorted(p.parent for p in DATA_ROOT.glob("*/chunks.jsonl"))
    model = load_model()
    collection = get_collection()
    for book_dir in book_dirs:
        if not (book_dir / "chunks.jsonl").exists():
            logger.error("chunks.jsonl 없음: %s (먼저 text_refine.py 실행)", book_dir)
            continue
        index_book(book_dir, collection, model)


if __name__ == "__main__":
    main()
