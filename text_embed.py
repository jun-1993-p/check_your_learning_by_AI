"""임베딩 공용: bge-m3 모델 불러오기·임베딩, chunks.jsonl 읽기, 블록 텍스트 변환.

벡터 저장소와 검색은 문단 단위로 text_paragraphs.py가 맡는다 (조각·소제목·부분 단위의 옛 임베딩은 폐기).
"""

import json
import logging
import os
import re
from pathlib import Path

from text_ingestion import DATA_ROOT

MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 2048  # 문단 단위는 이보다 훨씬 짧다 (bge-m3 최대 8192)
BATCH_SIZE = 8
CODE_MAX_LINES = 8  # 코드 비중이 벡터를 지배하지 않도록 블록당 앞부분만 넣는다
CODE_BLOCKS = ("code", "code_example")
IMAGE_PLACEHOLDER = re.compile(r"\[이미지: [^\]]*\]")
PROMPT_PREFIX = re.compile(r"^(>>>|\.\.\.) ?")

logger = logging.getLogger("text_embed")


def _code_for_embed(code: str) -> str:
    lines = [PROMPT_PREFIX.sub("", line) for line in code.split("\n")]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines[:CODE_MAX_LINES])


def block_to_text(block: dict) -> str:
    """임베딩용 블록 텍스트. 코드는 앞부분만, 이미지 표시는 뺀다."""
    kind = block["type"]
    if kind in CODE_BLOCKS:
        return _code_for_embed(block["code"])
    if kind == "table":
        rows = [block["header"], *block["rows"]] if block["header"] else block["rows"]
        return "\n".join(" | ".join(row) for row in rows)
    if kind == "concept_box":
        inner = "\n".join(block_to_text(b) for b in block["blocks"])
        return f"{block['title']}\n{inner}".strip()
    return IMAGE_PLACEHOLDER.sub("", block.get("text", "")).strip()


def render_block(block: dict) -> str:
    """LLM에 보여 줄 블록 텍스트. 임베딩용과 달리 코드·오류 출력까지 그대로 살린다."""
    kind = block["type"]
    if kind in CODE_BLOCKS:
        lang = block.get("lang") or ""
        text = f"```{lang}\n{block['code']}\n```"
        if block.get("error"):
            text += f"\n(오류 출력)\n{block['error']}"
        return text
    if kind == "table":
        rows = [block["header"], *block["rows"]] if block["header"] else block["rows"]
        return "\n".join(" | ".join(row) for row in rows)
    if kind == "concept_box":
        inner = "\n".join(render_block(b) for b in block["blocks"])
        return f"<{block['title']}>\n{inner}"
    if kind == "heading":
        return f"{'#' * block['level']} {block['text']}"
    if kind == "list":
        return f"- {block['text']}"
    return block.get("text", "")


def render_blocks(blocks: list[dict]) -> str:
    return "\n".join(t for b in blocks if (t := render_block(b).strip()))


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


def load_chunks(chunks_file: Path) -> list[dict]:
    chunks = []
    with chunks_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("%s %d번째 줄 무시: %s", chunks_file, line_no, e)
    return chunks


def load_chunk_index() -> dict[str, dict]:
    """모든 책의 chunks.jsonl을 chunk_id로 찾을 수 있게 읽는다 (문단의 원본 블록 조회용)."""
    index: dict[str, dict] = {}
    for chunks_file in sorted(DATA_ROOT.glob("*/chunks.jsonl")):
        for chunk in load_chunks(chunks_file):
            if chunk["chunk_id"] in index:
                logger.warning("chunk_id 중복: %s (%s)", chunk["chunk_id"], chunks_file)
            index[chunk["chunk_id"]] = chunk
    return index
