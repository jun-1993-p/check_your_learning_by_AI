"""문단 인덱스: 조각(chunks.jsonl)에서 A+ 문단을 만들고, bge-m3로 임베딩해 Chroma에 저장·검색한다.

설계: .idea_folder/정확도_2차_근거문장_인덱스_설계.md (0장). 문장·조각 단위 인덱스는 폐기했다.

A+ 문단 = 설명 블록(paragraph, note, key_point, caption, concept_box, 연속한 list)과 바로 뒤의
코드·실행 예시·표를 한 묶음(B)으로 보고, B를 같은 소제목 안에서 이웃과 합쳐 목표 길이(200자)까지
키운 것이다 (상한 600자, 100자 미만의 짧은 꼬리는 앞 단위에 붙인다). 제목 바로 뒤에 홀로 남은
코드·표는 뒤따르는 설명에 붙여서(없으면 앞 단위나 단독 단위로) 어떤 문단에도 안 들어가는 블록이 없게 한다.
제목(heading)과 소제목(subheading)은 문단 본문이 아니라 소제목 라벨과 URL 앵커로 쓰인다.
소제목이 없는 구간은 페이지 제목으로 대체한다. 한 페이지에서 그렇게 대체한 단위가 여럿이면
표시용 라벨을 "제목 - 1", "제목 - 2"로 구분한다. `[이미지: …]` 표시는 텍스트에서 빼고
`has_image`, `figure_ref`("그림 N" 참조) 플래그로만 남긴다.

저장:
    .data/vectorstore_paragraphs/           Chroma. 문단마다 벡터 하나 (embed_hash로 증분 갱신)
문단 자체는 파일로 저장하지 않는다. chunks.jsonl에서 필요할 때 만든다 (책 전체에 약 0.1초).
chunks.jsonl의 한 줄은 페이지 하나이고, 검색 단위(chunk)는 여기서 만드는 A+ 문단이다.

사용 예:
    python text_paragraphs.py                                  # 모든 책 증분 임베딩
    python text_paragraphs.py --book-id 1                      # 특정 책만
    python text_paragraphs.py --rebuild                        # 저장소를 archive로 옮기고 새로 생성
    python text_paragraphs.py --query "변수" --book-id 110     # 문단 검색
"""

import argparse
import functools
import hashlib
import logging
import os
import re
from collections import Counter
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
UPSERT_BATCH_SIZE = 64

EXPLAIN_BLOCKS = ("paragraph", "note", "key_point", "caption", "concept_box")
ATTACH_BLOCKS = (*CODE_BLOCKS, "table")  # 설명에 붙는 자료
MIN_UNIT_CHARS = 30
PLUS_TARGET_CHARS = 200  # 이 길이가 될 때까지 이웃 단위를 합친다
PLUS_MAX_CHARS = 600  # 합친 길이가 이를 넘으면 합치지 않는다
PLUS_TAIL_CHARS = 100  # 구간 끝에 남은 이보다 짧은 꼬리는 앞 단위에 붙인다
CORE_MARK = "<<핵심 문단>>"  # 페이지 맥락 안에서 핵심 문단이 있던 자리
PAGE_CONTEXT_CHARS = 4000  # 페이지 맥락 최대 글자 수
# "[이미지: 02_3_list.png]" 표시는 LLM에 정보가 없고 글자만 차지해서 뺀다 (뒤따르는 공백까지)
IMAGE_MARK = re.compile(r"\[이미지: [^\]]*\]\s*")
FIGURE_REF = re.compile(r"그림\s?\d+(\.\d+)*")  # 내용이 그림에 있다는 신호
FENCED_CODE = re.compile(r"```[^\n]*\n.*?```[ \t]*\n?", re.DOTALL)  # 렌더링된 텍스트의 코드 블록
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")

FETCH_LIMIT = 300  # 검색할 때 벡터에서 가져오는 후보 수 (관문을 거치고 나면 줄어든다)
GATE_STOPWORDS = {"자료형", "파이썬", "파이썬의", "값을", "저장하는", "공간", "기본"}
GATE_ALIASES = {
    "불리언": ["boolean", "bool"],
    "불": ["bool", "불 자료형"],
    "숫자형": ["숫자"],
}

# 적합성 관문: 문장 단위로 사실이 아닌 문장을 가려내는 규칙. 사실 문장이 MIN_FACT_CHARS보다 적으면 문단을 뺀다
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_FACT_CHARS = 40
# 정의 문장: "X는/이란 ... Y이다·의미한다·부른다·라고 한다"처럼 개념을 풀이하는 문장
DEFINITION = re.compile(
    r"(?:은|는|이란|란)\s.{4,}?"
    r"(?:의미한다|의미합니다|말한다|말합니다|뜻한다|뜻합니다|가리킨다|가리킵니다|"
    r"부른다|부릅니다|라고 한다|라고 합니다|이라고 한다|이라고 합니다)[.!]?$"
)  # '~것이다'·'~점입니다' 같은 서술문은 정의가 아니라서 이다·입니다는 넣지 않는다
# 코드가 있었던 문단(has_code)은 코드를 봐야 풀리는 문제가 나와 오답률이 높아 당분간 제외한다.
# False면 코드를 뺀 설명 글만으로 같은 기준을 적용해 통과시킨다
EXCLUDE_CODE = True
UNFIT_PATTERNS = {
    "도입·예고": re.compile(
        r"알아보(자|겠|도록)|살펴보(자|겠|도록)|배워 ?보(자|겠)|해 ?보(자|겠)|"
        r"다음 (장|절|절에서)|마무리|소개한다|시작해 ?보"
    ),
    "회고": re.compile(r"(배웠|살펴봤|알아봤|보았|봤)(습니다|다)|앞(에서|서)\s*(\S+\s*){0,3}(배웠|다뤘|설명)"),
    "비유": re.compile(
        r"비유|빗대|마치 |(인간|사람|현실|실생활|일상)(으로|에서|세계)\s*(치면|비유|는|에서는)|"
        r"드라마|유전형질|재산"
    ),
    "평가·감상": re.compile(r"매력|재미있|흥미|장벽|중요(하다|합니다)|어렵(게|다고) 느|걱정|두려"),
}

# 짧은 문단은 정보가 없어도 개념어가 겹치면 점수가 높게 나온다. 이보다 짧으면 길이에 비례해 감점한다
SHORT_UNIT_CHARS = 80
SHORT_PENALTY = 0.1  # 아주 짧은 문단(길이 0)에 빼는 최대 점수

logger = logging.getLogger("text_paragraphs")


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


def attach_blocks(
    blocks: list[dict], ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """설명 범위에 코드·실행 예시·표를 붙여서 어떤 문단에도 안 들어가는 블록이 없게 한다.

    1. 설명 바로 뒤에 이어지는 자료는 그 설명에 붙인다.
    2. 제목 바로 뒤처럼 홀로 남은 자료는 바로 뒤따르는 설명에 붙인다 (앞쪽으로 넓힌다).
    3. 뒤따르는 설명이 없으면, 사이에 소제목만 있는 바로 앞 단위에 붙인다.
    4. 그것도 없으면 단독 단위로 둔다 (MIN_UNIT_CHARS 미만이면 build_units에서 버린다).
    """
    result = []
    for start, end in ranges:
        while end < len(blocks) and blocks[end]["type"] in ATTACH_BLOCKS:
            end += 1
        result.append((start, end))
    covered = {i for start, end in result for i in range(start, end)}

    i = 0
    while i < len(blocks):
        if blocks[i]["type"] not in ATTACH_BLOCKS or i in covered:
            i += 1
            continue
        j = i
        while (
            j < len(blocks) and blocks[j]["type"] in ATTACH_BLOCKS and j not in covered
        ):
            j += 1
        following = next((k for k, (s, _) in enumerate(result) if s == j), None)
        previous = max(
            (
                k
                for k, (_, e) in enumerate(result)
                if e <= i
                and all(blocks[m]["type"] == "subheading" for m in range(e, i))
            ),
            key=lambda k: result[k][1],
            default=None,
        )
        if following is not None:
            result[following] = (i, result[following][1])
        elif previous is not None:
            result[previous] = (result[previous][0], j)
        else:
            result.append((i, j))
        i = j
    return sorted(result)


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


def code_ids_of(blocks: list[dict]) -> list[str]:
    """블록 목록의 코드 블록 id (개념 상자 안쪽 포함). 코드 본문은 code_blocks.jsonl에 있다."""
    ids: list[str] = []
    for block in blocks:
        if block["type"] in CODE_BLOCKS and block.get("code_id"):
            ids.append(block["code_id"])
        elif block["type"] == "concept_box":
            ids += code_ids_of(block["blocks"])
    return ids


def build_units(chunk: dict) -> list[dict]:
    """조각의 A+ 문단 목록. 각 문단은 블록 범위(block_start, block_end)를 갖는다.

    text에는 코드 블록이 없다 (code_ids/has_code로만 알린다). 임베딩 텍스트(embed_text)에는 코드가
    그대로 들어 있어서 벡터와 검색은 코드를 뺀 뒤에도 달라지지 않는다.
    """
    blocks = chunk["blocks"]
    secs = sections(blocks)
    ranges = merge_short(blocks, secs, attach_blocks(blocks, explain_ranges(blocks)))
    # 조각 path의 끝은 조각의 첫 소제목이라 페이지 제목이 아니다. 페이지 제목을 쓴다
    title = chunk.get("page_title") or (chunk.get("path") or [""])[-1]

    units: list[dict] = []
    for start, end in ranges:
        chosen = blocks[start:end]
        raw = render_blocks(chosen)
        full = IMAGE_MARK.sub("", raw).strip()
        if len(full) < MIN_UNIT_CHARS:  # 코드까지 센 길이로 거른다: 문단 목록과 id가 코드를 빼기 전과 같다
            continue
        # 코드는 문제 생성에 마이너스라 LLM에 주는 본문(text)에서 뺀다. 코드는 code_ids로 코드 저장소와 잇는다
        text = EXTRA_BLANK_LINES.sub("\n\n", FENCED_CODE.sub("", full)).strip()
        section, anchor = secs[start]
        body = embed_text(title, section, chosen)
        units.append(
            {
                # 블록 위치로 만든다: 앞에 문단이 끼어도 뒤 문단의 id가 밀리지 않는다
                "unit_id": f"{chunk['chunk_id']}#b{start:03d}",
                "chunk_id": chunk["chunk_id"],
                "page_id": chunk.get("page_id"),
                "book_id": chunk.get("book_id"),
                "path": chunk.get("path", []),
                "section": section or title,
                "fallback": not section,
                "url": anchor_url(chunk, anchor),
                "text": text,
                "code_ids": code_ids_of(chosen),
                "has_code": "```" in full,  # 목록 안에 들어간 코드처럼 code_id가 없는 코드도 센다
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


def get_collection(store_dir: Path = VECTORSTORE_DIR):
    # 순수 Python protobuf 구현으로 우회한다 (conda-forge chromadb의 opentelemetry-proto 호환 문제)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(store_dir),
        settings=Settings(anonymized_telemetry=False),  # 외부 전송 차단
    )
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embed_model": MODEL_NAME},
    )


def read_embeddings(store_dir: Path) -> dict[str, list[float]]:
    """저장소의 벡터를 임베딩 텍스트 해시로 찾을 수 있게 읽는다.

    --rebuild로 옛 저장소를 archive로 옮긴 뒤, 임베딩 텍스트가 그대로인 문단의 벡터를
    다시 계산하지 않고 재사용하는 데 쓴다. 텍스트가 같으면 벡터도 같다.
    """
    got = get_collection(store_dir).get(include=["embeddings", "metadatas"])
    return {
        m["embed_hash"]: list(map(float, vec))
        for vec, m in zip(got["embeddings"], got["metadatas"])
    }


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


def load_units(book_dir: Path) -> list[dict]:
    """책 폴더의 chunks.jsonl에서 A+ 문단을 만든다 (책 전체에 약 0.1초)."""
    return [u for c in load_chunks(book_dir / "chunks.jsonl") for u in build_units(c)]


def index_book(
    book_dir: Path,
    collection,
    get_model,
    reuse: dict[str, list[float]] | None = None,
) -> None:
    """chunks.jsonl → 문단 → 벡터 저장소. 임베딩 텍스트가 바뀐 문단만 다시 임베딩한다.

    get_model은 모델을 돌려주는 함수다. 새로 임베딩할 문단이 있을 때만 불러서,
    바뀐 게 없는 실행은 모델 로딩(약 13초)을 건너뛴다. reuse는 임베딩 텍스트 해시 →
    기존 벡터로, 있으면 모델을 거치지 않고 그 벡터를 쓴다 (read_embeddings 참고).
    """
    rows = load_units(book_dir)

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
    reuse = reuse or {}
    n_reuse = sum(r["embed_hash"] in reuse for r in todo)
    logger.info(
        "%s: 문단 %d개, 저장할 것 %d개 (벡터 재사용 %d, 새로 임베딩 %d)",
        book_dir.name,
        len(rows),
        len(todo),
        n_reuse,
        len(todo) - n_reuse,
    )
    if not todo:
        return
    model = None
    for start in range(0, len(todo), UPSERT_BATCH_SIZE):
        batch = todo[start : start + UPSERT_BATCH_SIZE]
        fresh = [r for r in batch if r["embed_hash"] not in reuse]
        if fresh:
            model = model or get_model()
            too_long = [
                r["unit_id"]
                for r in fresh
                if len(model.tokenizer(r["embed_text"])["input_ids"]) > MAX_SEQ_LENGTH
            ]
            if too_long:
                logger.warning(
                    "%d토큰 초과로 잘리는 문단: %s", MAX_SEQ_LENGTH, too_long
                )
            vectors = dict(
                zip(
                    (r["unit_id"] for r in fresh),
                    embed(model, [r["embed_text"] for r in fresh]),
                )
            )
        else:
            vectors = {}
        collection.upsert(
            ids=[r["unit_id"] for r in batch],
            embeddings=[
                vectors.get(r["unit_id"]) or reuse[r["embed_hash"]] for r in batch
            ],
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


def search_query(concept: str) -> str:
    """임베딩 질의. 본문이 영문으로 쓴 개념은 별칭을 붙인다. '불리언' → '불리언 (boolean, bool)'."""
    aliases = [a for w in re.split(r"[\s,]+", concept) for a in GATE_ALIASES.get(w, [])]
    return f"{concept} ({', '.join(aliases)})" if aliases else concept


def short_penalty(text: str) -> float:
    """SHORT_UNIT_CHARS보다 짧은 문단에 빼는 점수. 짧을수록 크다."""
    return SHORT_PENALTY * max(0.0, 1 - len(text.strip()) / SHORT_UNIT_CHARS)


def unfit_reason(text: str, has_code: bool = False) -> str | None:
    """빈칸 문제의 근거로 쓸 수 없는 문단이면 이유를, 쓸 수 있으면 None을 돌려준다.

    문단 안에서 도입·예고·회고, 비유, 평가·감상 문장을 빼고 남는 '사실' 문장이
    너무 적으면 부적합으로 본다. text에는 코드가 없으므로 코드가 있었던 문단(has_code)은
    EXCLUDE_CODE이 True면(기본) 제외하고, False면 설명 글만으로 같은 기준을 적용한다.
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    # 첫 문장이 도입·회고면 뒤 문장이 앞 예제·다음 코드를 가리키는 경우가 많아 문단째 뺀다.
    # 단, 뒤에 스스로 선 정의 문장("X는 Y이다")이 있으면 그 문장으로 낼 수 있어 남긴다
    for reason in ("도입·예고", "회고"):
        if sentences and UNFIT_PATTERNS[reason].search(sentences[0]):
            if any(
                DEFINITION.search(s) and not any(p.search(s) for p in UNFIT_PATTERNS.values())
                for s in sentences[1:]
            ):
                break
            return f"{reason}(첫 문장)"
    if has_code and EXCLUDE_CODE:
        return "코드 포함"
    facts = 0
    reasons: Counter = Counter()
    for sentence in sentences:
        for reason, pattern in UNFIT_PATTERNS.items():
            if pattern.search(sentence):
                reasons[reason] += 1
                break
        else:
            facts += len(sentence)
    if facts >= MIN_FACT_CHARS:
        return None
    return reasons.most_common(1)[0][0] if reasons else "내용 부족"


def fit_gate(hits: list[dict]) -> list[dict]:
    """unfit_reason에 걸리지 않는 문단만 남긴다. 걸러 낸 문단은 DEBUG 로그에 이유와 함께 남는다."""
    kept = []
    for hit in hits:
        reason = unfit_reason(hit["text"], hit.get("has_code", False))
        if reason is None:
            kept.append(hit)
        else:
            logger.debug("부적합 문단(%s): %s", reason, hit["text"][:40])
    return kept


def page_gate(concept: str, hits: list[dict]) -> list[dict]:
    """개념 핵심어가 페이지 경로에 있는 문단만 남긴다. 하나도 없으면 그대로 둔다."""
    tokens = concept_tokens(concept)
    kept = [h for h in hits if any(t in " > ".join(h["path"]).lower() for t in tokens)]
    return kept or hits


class ParagraphIndex:
    """개념·질문에 맞는 문단을 점수순으로 찾는다. 문단은 책별 chunks.jsonl에서 그때그때 만든다."""

    def __init__(self, model=None, collection=None, chunk_index=None):
        self.model = model or load_model()
        self.collection = collection or get_collection()
        self._chunk_index = chunk_index
        self._rows: dict[str, dict[str, dict]] = {}

    @property
    def chunk_index(self) -> dict[str, dict]:
        if self._chunk_index is None:
            self._chunk_index = load_chunk_index()
        return self._chunk_index

    def rows_in(self, source_dir: str) -> dict[str, dict]:
        """폴더 이름(book1, unbound 등)으로 찾는다. book_id가 없는 책도 같은 방식으로 읽는다."""
        if source_dir not in self._rows:
            book_dir = DATA_ROOT / source_dir
            if not (book_dir / "chunks.jsonl").exists():
                raise FileNotFoundError(f"{book_dir / 'chunks.jsonl'} 없음 (먼저 정제)")
            self._rows[source_dir] = {u["unit_id"]: u for u in load_units(book_dir)}
        return self._rows[source_dir]

    def rows(self, book_id: int | None) -> dict[str, dict]:
        return self.rows_in(Paths.for_book(book_id).book_dir.name)

    def find(
        self,
        concept: str,
        book_id: int | None = None,
        exclude: frozenset[str] = frozenset(),
        fit: bool = False,
    ) -> list[dict]:
        """문단을 점수순으로. 점수는 코사인 유사도에서 짧은 문단 감점을 뺀 값이다.

        관문 순서: 페이지 관문 → (fit=True면) 적합성 관문 → 점수순 정렬. exclude는 이미 쓴 unit_id.
        """
        query = embed(self.model, [search_query(concept)])
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
            row = self.rows_in(meta["source_dir"]).get(unit_id)
            if row is None:  # 규칙이 바뀌어 남은 옛 벡터
                continue
            if row["embed_hash"] != meta.get("embed_hash"):
                continue
            score = 1 - dist - short_penalty(row["text"])
            found.append({**row, "score": round(score, 4), "distance": dist})
        found = page_gate(concept, found)
        if fit:
            found = fit_gate(found)
        return sorted(found, key=lambda u: -u["score"])

    def in_page(self, page_id: int, book_id: int | None, query: str) -> list[dict]:
        """한 페이지의 문단 전부를 질의와의 유사도순으로. 페이지 관문은 거치지 않는다.

        학습자가 읽은 페이지의 다른 문단을 문제 후보로 고를 때 쓴다.
        """
        units = [u for u in self.rows(book_id).values() if u["page_id"] == page_id]
        if not units:
            return []
        vec = np.asarray(embed(self.model, [query])[0], dtype=np.float32)
        scores = self.vectors([u["unit_id"] for u in units]) @ vec
        scored = [
            {**u, "score": round(float(s), 4), "distance": 1 - float(s)}
            for u, s in zip(units, scores)
        ]
        return sorted(scored, key=lambda u: -u["score"])

    def page_context(self, unit: dict, limit: int = PAGE_CONTEXT_CHARS) -> str:
        """핵심 문단이 속한 페이지(조각). 핵심 문단은 CORE_MARK 한 줄로 바꿔 중복해서 넣지 않는다.

        길이가 limit를 넘으면 핵심 문단에서 앞뒤로 블록을 번갈아 넓혀 가며 limit 안에서 자른다.
        """
        blocks = self.chunk_index[unit["chunk_id"]]["blocks"]
        # 코드는 문제 생성에 마이너스라 맥락에서도 뺀다 (코드 블록은 빈 문자열이 되어 건너뛴다)
        rendered = [
            FENCED_CODE.sub("", IMAGE_MARK.sub("", render_block(b))).strip() for b in blocks
        ]
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
        help="저장소를 archive로 옮기고 새로 생성 (임베딩 텍스트가 같은 문단은 벡터를 재사용)",
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

    reuse = None
    if args.rebuild:
        archived = move_to_archive(VECTORSTORE_DIR)
        reuse = read_embeddings(archived) if archived else None
    if args.book_id is not None:
        book_dirs = [Paths.for_book(args.book_id).book_dir]
    else:
        book_dirs = sorted(p.parent for p in DATA_ROOT.glob("*/chunks.jsonl"))
    get_model = functools.cache(load_model)  # 책이 여러 권이어도 한 번만 불러온다
    collection = get_collection()
    for book_dir in book_dirs:
        if not (book_dir / "chunks.jsonl").exists():
            logger.error("chunks.jsonl 없음: %s (먼저 text_refine.py 실행)", book_dir)
            continue
        index_book(book_dir, collection, get_model, reuse)


if __name__ == "__main__":
    main()
