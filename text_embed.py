"""chunks.jsonl의 개념 조각을 bge-m3로 임베딩해 Chroma에 저장한다.

조각(chunk) 단위 벡터와 소제목(child) 단위 벡터를 함께 저장하고,
검색 결과는 부모 조각 단위로 묶어서 돌려준다.
조각(또는 소제목 구간)이 길면 부분(part) 벡터를 더한다. 정제 단계에서 조각을 다시 나누지 않고도
긴 조각에서 질의와 가까운 블록 범위만 자료로 쓸 수 있게 한다(조각 순위에는 쓰지 않는다).

사용 예:
    python text_embed.py                       # 모든 책 증분 임베딩
    python text_embed.py --book-id 1           # 특정 책만
    python text_embed.py --rebuild             # 기존 저장소를 archive로 옮기고 새로 생성
    python text_embed.py --query "문자열 공백 제거" --k 5
    python text_embed.py --query "슬라이싱" --page-id 13
"""

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from text_ingestion import DATA_ROOT, Paths, move_to_archive

MODEL_NAME = "BAAI/bge-m3"
VECTORSTORE_DIR = DATA_ROOT / "vectorstore"
COLLECTION_NAME = "chunks_bge-m3"
MAX_SEQ_LENGTH = 2048  # 가장 긴 조각도 이 안에 들어간다 (bge-m3 최대 8192)
BATCH_SIZE = 8
CODE_MAX_LINES = 8  # 코드 비중이 벡터를 지배하지 않도록 블록당 앞부분만 넣는다
PART_MAX_CHARS = 2000  # 임베딩 텍스트가 이보다 긴 구간만 부분으로 나눈다
PART_TARGET_CHARS = 1500  # 부분 하나의 목표 길이
PART_MIN_CHARS = 400  # 목표 길이로 끊다 남은 꼬리가 이보다 짧으면 앞 부분에 붙인다
CODE_BLOCKS = ("code", "code_example")
SEARCH_FETCH_FACTOR = (
    8  # 한 조각의 여러 단위(조각·소제목·부분)가 겹치므로 k의 배수로 가져온다
)
IMAGE_PLACEHOLDER = re.compile(r"\[이미지: [^\]]*\]")
PROMPT_PREFIX = re.compile(r"^(>>>|\.\.\.) ?")

logger = logging.getLogger("text_embed")


@dataclass
class Unit:
    """벡터 하나에 대응하는 단위. 조각 전체(chunk) 또는 소제목 단위(child)."""

    id: str
    chunk_id: str
    kind: str
    heading: str
    text: str
    metadata: dict


# ---------------------------------------------------------------- 임베딩 텍스트


def _code_for_embed(code: str) -> str:
    lines = [PROMPT_PREFIX.sub("", line) for line in code.split("\n")]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines[:CODE_MAX_LINES])


def block_to_text(block: dict) -> str:
    kind = block["type"]
    if kind in ("code", "code_example"):
        return _code_for_embed(block["code"])
    if kind == "table":
        rows = [block["header"], *block["rows"]] if block["header"] else block["rows"]
        return "\n".join(" | ".join(row) for row in rows)
    if kind == "concept_box":
        inner = "\n".join(block_to_text(b) for b in block["blocks"])
        return f"{block['title']}\n{inner}".strip()
    return IMAGE_PLACEHOLDER.sub("", block.get("text", "")).strip()


def build_embed_text(path: list[str], headings: list[str], blocks: list[dict]) -> str:
    parts = [f"[{' > '.join(path)}]"]
    if headings:
        parts.append("다루는 내용: " + ", ".join(headings))
    parts.extend(t for b in blocks if (t := block_to_text(b)))
    return "\n".join(parts)


def split_children(blocks: list[dict]) -> list[tuple[str, list[dict]]]:
    """h2/h3 제목 기준으로 나눈다. 제목 앞 도입부는 첫 소제목에 붙인다."""
    children: list[tuple[str, list[dict]]] = []
    intro: list[dict] = []
    for block in blocks:
        if block["type"] == "heading" and block["level"] in (2, 3):
            children.append((block["text"], [*intro, block]))
            intro = []
        elif children:
            children[-1][1].append(block)
        else:
            intro.append(block)
    return children


def child_ranges(blocks: list[dict]) -> list[tuple[str, int, int]]:
    """split_children과 같은 기준의 (소제목, 시작, 끝) 블록 범위. 도입부는 첫 소제목에 붙인다."""
    starts = [
        n
        for n, b in enumerate(blocks)
        if b["type"] == "heading" and b["level"] in (2, 3)
    ]
    return [
        (
            blocks[s]["text"],
            0 if i == 0 else s,
            starts[i + 1] if i + 1 < len(starts) else len(blocks),
        )
        for i, s in enumerate(starts)
    ]


def _block_groups(blocks: list[dict], start: int, end: int) -> list[tuple[int, int]]:
    """끊을 수 있는 최소 묶음. 설명 블록과 바로 뒤의 코드·실행 예시는 한 묶음이다."""
    groups: list[list[int]] = []
    for n in range(start, end):
        if blocks[n]["type"] in CODE_BLOCKS and groups:
            groups[-1][1] = n + 1
        else:
            groups.append([n, n + 1])
    return [(s, e) for s, e in groups]


def part_ranges(blocks: list[dict], start: int, end: int) -> list[tuple[int, int]]:
    """[start, end) 구간이 길면 부분 범위들로 나눈다. 나눌 필요가 없으면 빈 목록.

    subheading 경계에서 먼저 나누고, 그래도 긴 구간은 묶음을 쌓아 목표 길이에서 끊는다.
    """

    def length(s: int, e: int) -> int:
        return sum(len(block_to_text(b)) for b in blocks[s:e])

    if length(start, end) <= PART_MAX_CHARS:
        return []
    cuts = [n for n in range(start + 1, end) if blocks[n]["type"] == "subheading"]
    segments = list(zip([start, *cuts], [*cuts, end]))
    parts: list[tuple[int, int]] = []
    for seg_start, seg_end in segments:
        if length(seg_start, seg_end) <= PART_MAX_CHARS:
            parts.append((seg_start, seg_end))
            continue
        packed: list[tuple[int, int]] = []
        cur_start, cur_len = seg_start, 0
        for g_start, g_end in _block_groups(blocks, seg_start, seg_end):
            g_len = length(g_start, g_end)
            if cur_len and cur_len + g_len > PART_TARGET_CHARS:
                packed.append((cur_start, g_start))
                cur_start, cur_len = g_start, 0
            cur_len += g_len
        if packed and cur_len < PART_MIN_CHARS:  # 짧은 꼬리는 앞 부분에 붙인다
            cur_start = packed.pop()[0]
        packed.append((cur_start, seg_end))
        parts += packed
    return parts if len(parts) > 1 else []


def _metadata(chunk: dict, kind: str, heading: str, text: str) -> dict:
    meta = {
        "chunk_id": chunk["chunk_id"],
        "kind": kind,
        "heading": heading,
        "page_id": chunk["page_id"],
        "book_id": chunk.get("book_id"),
        "page_title": chunk.get("page_title"),
        "path": " > ".join(chunk.get("path", [])),
        "source_url": chunk.get("source_url"),
        "image_dependent": chunk.get("image_dependent", False),
        "has_error_example": chunk.get("has_error_example", False),
        "embed_model": MODEL_NAME,
        "embed_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return {k: v for k, v in meta.items() if v is not None}  # Chroma는 None 불가


def build_units(chunk: dict) -> list[Unit]:
    path = chunk.get("path", [])
    text = build_embed_text(path, chunk.get("headings", []), chunk["blocks"])
    units = [
        Unit(
            id=f"{chunk['chunk_id']}#chunk",
            chunk_id=chunk["chunk_id"],
            kind="chunk",
            heading=" / ".join(chunk.get("headings", [])),
            text=text,
            metadata=_metadata(chunk, "chunk", "", text),
        )
    ]
    blocks = chunk["blocks"]
    children = split_children(blocks)
    if len(children) >= 2:  # 소제목이 하나뿐이면 조각 벡터와 같다
        for n, (heading, child_blocks) in enumerate(children, start=1):
            child_text = build_embed_text([*path, heading], [], child_blocks)
            units.append(
                Unit(
                    id=f"{chunk['chunk_id']}#c{n:02d}",
                    chunk_id=chunk["chunk_id"],
                    kind="child",
                    heading=heading,
                    text=child_text,
                    metadata=_metadata(chunk, "child", heading, child_text),
                )
            )
        regions = child_ranges(blocks)
    else:
        regions = [("", 0, len(blocks))]

    n_part = 0
    for heading, start, end in regions:
        for part_start, part_end in part_ranges(blocks, start, end):
            n_part += 1
            sub = blocks[part_start]
            part_heading = sub["text"] if sub["type"] == "subheading" else heading
            part_path = [*path, heading] if heading else path
            part_text = build_embed_text(part_path, [], blocks[part_start:part_end])
            units.append(
                Unit(
                    id=f"{chunk['chunk_id']}#p{n_part:02d}",
                    chunk_id=chunk["chunk_id"],
                    kind="part",
                    heading=part_heading,
                    text=part_text,
                    metadata={
                        **_metadata(chunk, "part", part_heading, part_text),
                        "block_start": part_start,
                        "block_end": part_end,
                    },
                )
            )
    return units


# ---------------------------------------------------------------- 모델·저장소


def load_model():
    # 캐시된 모델만 사용한다 (외부 요청 방지). 받으려면 .env에 HF_HUB_OFFLINE=0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    device = os.getenv("EMBED_DEVICE", "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed(model, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > BATCH_SIZE,
    )
    return vectors.tolist()


def get_collection():
    # conda-forge chromadb가 옛 opentelemetry-proto(protoc 구버전 생성 코드)에 묶여 있어
    # protobuf 6에서 import 오류가 난다. 순수 Python 구현으로 우회한다.
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


# ---------------------------------------------------------------- 실행


def load_chunks(chunks_file: Path) -> list[dict]:
    chunks = []
    with chunks_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("%s %d번째 줄 무시: %s", chunks_file, line_no, e)
    return chunks


def index_book(book_dir: Path, collection, model) -> None:
    chunks_file = book_dir / "chunks.jsonl"
    units = [u for c in load_chunks(chunks_file) for u in build_units(c)]
    # book_id가 없는 데이터(unbound, 구버전 pages.jsonl)도 있으므로
    # 저장 폴더 이름(book1, unbound 등)으로 기존 벡터를 찾는다
    for u in units:
        u.metadata["source_dir"] = book_dir.name

    got = collection.get(where={"source_dir": book_dir.name}, include=["metadatas"])
    existing = {
        i: m.get("embed_hash", "") for i, m in zip(got["ids"], got["metadatas"])
    }

    todo = [u for u in units if existing.get(u.id) != u.metadata["embed_hash"]]
    stale = set(existing) - {u.id for u in units}
    if stale:
        logger.warning(
            "%s: 더 이상 없는 벡터 %d개가 남아 있음. --rebuild로 정리하세요",
            book_dir.name,
            len(stale),
        )
    logger.info(
        "%s: 단위 %d개 (조각 %d, 소제목 %d, 부분 %d), 새로 임베딩 %d개",
        book_dir.name,
        len(units),
        sum(u.kind == "chunk" for u in units),
        sum(u.kind == "child" for u in units),
        sum(u.kind == "part" for u in units),
        len(todo),
    )
    if not todo:
        return

    too_long = [
        u.id for u in todo if len(model.tokenizer(u.text)["input_ids"]) > MAX_SEQ_LENGTH
    ]
    if too_long:
        logger.warning("%d토큰 초과로 잘리는 단위: %s", MAX_SEQ_LENGTH, too_long)

    for start in range(0, len(todo), 64):
        batch = todo[start : start + 64]
        collection.upsert(
            ids=[u.id for u in batch],
            embeddings=embed(model, [u.text for u in batch]),
            documents=[u.text for u in batch],
            metadatas=[u.metadata for u in batch],
        )
        logger.info("upsert %d/%d", start + len(batch), len(todo))


def search(
    query: str,
    k: int = 5,
    book_id: int | None = None,
    page_id: int | None = None,
    model=None,
    collection=None,
) -> list[dict]:
    """질의와 가까운 조각을 찾는다. 소제목 벡터가 걸려도 부모 조각으로 묶는다.

    조각 순위는 조각·소제목 벡터로만 정한다. 부분(part) 벡터는 짧고 코드 비중이 커서
    짧은 질의("리스트")에 다른 페이지의 코드 부분이 끼어들기 때문이다. 부분 벡터는
    고른 조각 안에서 가장 가까운 블록 범위(block_start, block_end)를 찾는 데만 쓴다.
    """
    model = model or load_model()
    collection = collection or get_collection()
    filters = [
        {key: value}
        for key, value in (("book_id", book_id), ("page_id", page_id))
        if value is not None
    ]
    query_vec = embed(model, [query])
    result = collection.query(
        query_embeddings=query_vec,
        n_results=k * SEARCH_FETCH_FACTOR,  # 같은 조각의 여러 단위가 겹치므로 넉넉히
        where=_where([*filters, {"kind": {"$ne": "part"}}]),
        include=["metadatas", "distances"],
    )
    best: dict[str, dict] = {}
    for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
        chunk_id = meta["chunk_id"]
        if chunk_id not in best or dist < best[chunk_id]["distance"]:
            best[chunk_id] = {
                "chunk_id": chunk_id,
                "distance": dist,
                "matched": meta["kind"],
                "heading": meta.get("heading", ""),
                "path": meta.get("path", ""),
                "source_url": meta.get("source_url", ""),
                "block_start": None,
                "block_end": None,
            }
    hits = sorted(best.values(), key=lambda r: r["distance"])[:k]
    if not hits:
        return hits

    by_id = {h["chunk_id"]: h for h in hits}
    parts = collection.query(
        query_embeddings=query_vec,
        n_results=k * SEARCH_FETCH_FACTOR,
        where=_where([{"kind": "part"}, {"chunk_id": {"$in": list(by_id)}}]),
        include=["metadatas"],
    )
    for meta in parts["metadatas"][
        0
    ]:  # 거리순이라 조각마다 처음 나온 부분이 가장 가깝다
        hit = by_id[meta["chunk_id"]]
        if hit["block_start"] is None:
            hit["block_start"] = meta["block_start"]
            hit["block_end"] = meta["block_end"]
            hit["part_heading"] = meta.get("heading", "")
    return hits


def _where(filters: list[dict]) -> dict | None:
    if not filters:
        return None
    return filters[0] if len(filters) == 1 else {"$and": filters}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="개념 조각 임베딩·검색 (bge-m3 + Chroma)"
    )
    parser.add_argument("--book-id", type=int, default=None, help="생략하면 모든 책")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="기존 저장소를 archive로 옮기고 새로 생성",
    )
    parser.add_argument("--query", help="임베딩 대신 검색만 실행")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--page-id", type=int, default=None, help="검색 범위 제한")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.query:
        results = search(args.query, args.k, args.book_id, args.page_id)
        for r in results:
            print(
                f"{r['distance']:.3f} {r['chunk_id']:>9} [{r['matched']}] {r['path']}"
            )
            if r["matched"] == "child":
                print(f"{'':>16}↳ {r['heading']}")
            if r["block_start"] is not None:
                print(
                    f"{'':>16}↳ 가까운 부분: 블록 {r['block_start']}~{r['block_end']}"
                    f" {r.get('part_heading', '')}"
                )
            print(f"{'':>16}{r['source_url']}")
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
