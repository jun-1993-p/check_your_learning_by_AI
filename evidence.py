"""문장 단위 근거 인덱스: 개념의 근거 문장을 고르고, 문장이 책에 실제로 있는지 확인한다.

설계: .idea_folder/정확도_2차_근거문장_인덱스_설계.md

임베딩 전에는 문장이 아니거나 깨진 것만 거른다. 문맥 의존·비유·잡담처럼 읽고 판단해야
하는 문장은 문제를 만드는 LLM이 거부한다. 근거 문장은 개수를 고정하지 않고 점수순으로 돌려준다.

사용 예:
    python evidence.py                                  # 모든 책 문장 인덱스 증분 갱신
    python evidence.py --book-id 1                      # 특정 책만
    python evidence.py --rebuild                        # 문장 저장소를 archive로 옮기고 새로 생성
    python evidence.py --query "딕셔너리 자료형" --book-id 1
"""

import argparse
import builtins
import hashlib
import json
import keyword
import logging
import os
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import text_answer as ta
from text_embed import (
    IMAGE_PLACEHOLDER,
    MODEL_NAME,
    get_collection,
    load_chunks,
    load_model,
    search,
)
from text_ingestion import DATA_ROOT, Paths, move_to_archive
from text_refine import write_if_changed

SENTENCE_STORE_DIR = DATA_ROOT / "vectorstore_sentences"
SENTENCE_COLLECTION = "sentences_bge-m3"
SENTENCES_FILE = "sentences.jsonl"
ENCODE_BATCH_SIZE = 32  # 문장은 짧아서 조각 임베딩(8)보다 크게 잡는다
UPSERT_BATCH_SIZE = 256

TEXT_BLOCKS = ("paragraph", "note", "list", "key_point", "caption")
SECTION_BLOCKS = ("subheading",)
# 문장 끝(~다./~요.) 바로 뒤에 한글이 붙은 경우도 나눈다. `클래스_이름.클래스변수` 같은 코드는 나누지 않는다
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[다요][.!?])(?=[가-힣])")

# ---------------------------------------------------------------- 임베딩 전 규칙
# 문장이 아니거나 깨진 것, 떼면 뜻이 바뀌는 것만 거른다 (판단이 필요 없는 규칙)
MIN_CHARS = 15
PRE_RULES = [
    ("코드 펜스", re.compile(r"```")),
    # "그림 2.3 파이썬 바인딩의 의미" (뒤에 조사가 붙은 "그림 2.5와 같이"는 그림 참조)
    (
        "캡션",
        re.compile(r"^(그림|예제|표)\s?\d+(\.\d+)*(?![\d.와과처에을를의은는이가])"),
    ),
    # 내용이 그림·표·예제 코드에 있고, 떼면 뜻이 바뀐다
    ("그림·표·예제 참조", re.compile(r"(그림|표|예제)\s?\d+(\.\d+)*")),
    ("질문문", re.compile(r"\?[\"'”’)]?$")),
]
QUOTES = re.compile(r"[\"“”]")
NO_SPACE_RUN = re.compile(r"\S{15,}")
HANGUL_RUN = re.compile(r"[가-힣]{6,}")

# ---------------------------------------------------------------- 예제 값 (순위 감점)
GENERIC_NAMES = (
    set(keyword.kwlist)
    | set(dir(builtins))
    | {m for t in (str, list, dict, set, tuple, int, float) for m in dir(t)}
    | {"self", "cls", "python", "print", "True", "False", "None"}
    | {"dict_keys", "dict_values", "dict_items", "NoneType", "function", "method"}
)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STRING_LIT = re.compile(r"""['"]([A-Za-z0-9가-힣_][^'"\n]{1,29})['"]""")  # 2글자 이상
NUMBER = re.compile(r"\d[\d,]{2,}")

# ---------------------------------------------------------------- 검색·순위
CHUNK_K = 5
DELTA = 0.12  # 감점 전 점수가 개념별 중앙값 - DELTA 미만이면 뺀다
EXAMPLE_PENALTY = 0.15
MIN_SCORE = 0.45
DUP_SIM = 0.85
# 개념 이름·소제목을 그대로 담아 관련도는 높지만 LLM이 거부할 문장 형태. 빼지 않고 뒤로 보낸다
DEMOTED_FORMS = [
    # 안내: "~살펴보자", "응용해 보겠습니다", "알아본다", "살펴보는 것이 가장 알기 쉽다"
    re.compile(
        r"(보자|봅시다|보겠습니다|알아본다|알아봅니다|배우겠습니다)[.!]?$|살펴보는\s것이"
    ),
    # 예고: "다음은 …이다", "다음 표는", "다음과 같다/같이", "다음처럼"
    re.compile(r"^다음은\s.*(이다|입니다)\.?$|다음과\s같|^다음\s표는|다음처럼"),
    # 예시 지칭: "~예이다", "위 예제는", "위 예와 같이", "위와 같은 상황에서", "~확인할 수 있다"
    re.compile(
        r"예(이다|입니다)\.?$|위\s?(예제|예|코드)(는|에서|와\s같이|처럼)|위와\s같은"
        r"|확인할\s수\s있(다|습니다)\.?$"
    ),
    # 메타: "이번 장/절에서", "6.1절에서", "앞 절에서", "앞서 설명한 것처럼"
    re.compile(
        r"^(이번|이)\s?(장|절)에서|\d+(\.\d+)?\s?(장|절)에서|앞\s절에서"
        r"|(앞서|앞에서)\s(설명한|살펴본|배운)\s것처럼"
    ),
]
GATE_STOPWORDS = {"자료형", "파이썬", "파이썬의", "값을", "저장하는", "공간", "기본"}
GATE_ALIASES = {
    "불리언": ["boolean", "bool"],
    "불": ["bool", "불 자료형"],
    "숫자형": ["숫자"],
}

logger = logging.getLogger("evidence")


# ---------------------------------------------------------------- 문장 추출


def _walk(blocks: list[dict]):
    for block in blocks:
        if block["type"] == "concept_box":
            yield from _walk(block["blocks"])
        else:
            yield block


def drop_reason(text: str) -> str:
    for name, rx in PRE_RULES:
        if rx.search(text):
            return name
    if len(QUOTES.findall(text)) % 2:
        return "따옴표 짝 안 맞음"
    if any(HANGUL_RUN.search(run) for run in NO_SPACE_RUN.findall(text)):
        return "띄어쓰기 깨짐"
    if len(text) < MIN_CHARS:
        return f"{MIN_CHARS}자 미만"
    return ""


def chunk_sentences(chunk: dict) -> list[tuple[str, str]]:
    """조각의 (소제목, 문장) 목록. 소제목이 없으면 페이지 제목을 쓴다."""
    section = chunk["page_title"]
    result = []
    for block in _walk(chunk["blocks"]):
        kind = block["type"]
        if (
            kind == "heading" and block.get("level") in (2, 3)
        ) or kind in SECTION_BLOCKS:
            section = block.get("text", "").strip() or section
            continue
        if kind not in TEXT_BLOCKS:
            continue
        text = IMAGE_PLACEHOLDER.sub("", block.get("text", ""))
        result += [
            (section, s) for s in map(str.strip, SENTENCE_SPLIT.split(text)) if s
        ]
    return result


def page_example_terms(page_chunks: list[dict]) -> set[str]:
    """페이지 코드에 나온 예제 고유 이름·문자열 값·3자리 이상 숫자.

    키워드·내장 이름·내장 타입 메서드와 소제목에 나온 이름(설명 대상)은 뺀다.
    """
    headings = " ".join(
        b.get("text", "")
        for c in page_chunks
        for b in _walk(c["blocks"])
        if b["type"] in ("heading", "subheading")
    ).lower()
    terms: set[str] = set()
    for chunk in page_chunks:
        for block in _walk(chunk["blocks"]):
            if block["type"] not in ("code", "code_example"):
                continue
            code = block.get("code", "")
            terms.update(lit.strip() for lit in STRING_LIT.findall(code))
            for name in IDENT.findall(STRING_LIT.sub(" ", code)):
                has_digit = any(ch.isdigit() for ch in name)
                if (len(name) >= 3 or (len(name) == 2 and has_digit)) and (
                    name.lower() not in headings
                ):
                    terms.add(name)
            terms.update(num.replace(",", "") for num in NUMBER.findall(code))
    return {t for t in terms if t and t not in GENERIC_NAMES}


def example_terms_in(sentence: str, terms: set[str]) -> list[str]:
    plain = sentence.replace(",", "")
    hits = []
    for term in terms:
        if term.isdigit():
            if re.search(rf"(?<!\d){term}(?!\d)", plain):
                hits.append(term)
        elif re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", sentence
        ):
            hits.append(term)
    return sorted(hits)


def build_rows(chunks: list[dict]) -> list[dict]:
    by_page: dict[int, list[dict]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk["page_id"], []).append(chunk)
    page_terms = {pid: page_example_terms(cs) for pid, cs in by_page.items()}
    rows = []
    for chunk in chunks:
        for n, (section, text) in enumerate(chunk_sentences(chunk), start=1):
            reason = drop_reason(text)
            rows.append(
                {
                    "sentence_id": f"{chunk['chunk_id']}#s{n:03d}",
                    "chunk_id": chunk["chunk_id"],
                    "page_id": chunk["page_id"],
                    "book_id": chunk.get("book_id"),
                    "path": " > ".join(chunk.get("path", [])),
                    "source_url": chunk.get("source_url", ""),
                    "section": section,
                    "text": text,
                    "drop_reason": reason,
                    "example_terms": []
                    if reason
                    else example_terms_in(text, page_terms[chunk["page_id"]]),
                }
            )
    return rows


def embed_hash(row: dict) -> str:
    return hashlib.sha256(f"{row['section']}\n{row['text']}".encode()).hexdigest()


# ---------------------------------------------------------------- 저장소


def get_sentence_collection():
    # text_embed.get_collection과 같은 이유로 순수 Python protobuf 구현을 쓴다
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(SENTENCE_STORE_DIR),
        settings=Settings(anonymized_telemetry=False),  # 외부 전송 차단
    )
    return client.get_or_create_collection(
        SENTENCE_COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": MODEL_NAME},
    )


def encode(model, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > ENCODE_BATCH_SIZE * 4,
    )


def index_book(book_dir: Path, collection, model) -> None:
    rows = build_rows(load_chunks(book_dir / "chunks.jsonl"))
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if write_if_changed(book_dir / SENTENCES_FILE, content):
        logger.info("%s: %s 갱신", book_dir.name, SENTENCES_FILE)
    reasons = Counter(r["drop_reason"] or "통과" for r in rows)
    logger.info(
        "%s: 문장 %d개 %s", book_dir.name, len(rows), dict(reasons.most_common())
    )

    kept = [r for r in rows if not r["drop_reason"]]
    got = collection.get(where={"source_dir": book_dir.name}, include=["metadatas"])
    existing = {
        i: m.get("embed_hash", "") for i, m in zip(got["ids"], got["metadatas"])
    }
    todo = [r for r in kept if existing.get(r["sentence_id"]) != embed_hash(r)]
    stale = set(existing) - {r["sentence_id"] for r in kept}
    if stale:
        # 지우지 않는다. 검색할 때 sentences.jsonl의 통과 문장과 교집합만 쓴다
        logger.warning(
            "%s: 더 이상 쓰지 않는 벡터 %d개가 남아 있음. --rebuild로 정리하세요",
            book_dir.name,
            len(stale),
        )
    logger.info("%s: 통과 %d개, 새로 임베딩 %d개", book_dir.name, len(kept), len(todo))
    if not todo:
        return

    sections = sorted({r["section"] for r in todo})
    section_vecs = dict(zip(sections, encode(model, sections)))
    for start in range(0, len(todo), UPSERT_BATCH_SIZE):
        batch = todo[start : start + UPSERT_BATCH_SIZE]
        vecs = encode(model, [r["text"] for r in batch])
        collection.upsert(
            ids=[r["sentence_id"] for r in batch],
            embeddings=vecs.tolist(),
            documents=[r["text"] for r in batch],
            metadatas=[
                {
                    "sentence_id": r["sentence_id"],
                    "chunk_id": r["chunk_id"],
                    "page_id": r["page_id"],
                    "book_id": r["book_id"] if r["book_id"] is not None else -1,
                    "section": r["section"],
                    "source_dir": book_dir.name,
                    "example_terms": ",".join(r["example_terms"]),
                    "section_sim": float(v @ section_vecs[r["section"]]),
                    "embed_model": MODEL_NAME,
                    "embed_hash": embed_hash(r),
                }
                for r, v in zip(batch, vecs)
            ],
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
    """개념 핵심어가 페이지 경로에 있는 조각만 남긴다. 하나도 없으면 그대로 둔다."""
    tokens = concept_tokens(concept)
    kept = [h for h in hits if any(t in h.get("path", "").lower() for t in tokens)]
    for h in hits:
        if h not in kept:
            logger.debug("페이지 관문 탈락 %s: %s", tokens, h.get("path"))
    return kept or hits


def is_demoted_form(text: str) -> bool:
    return any(rx.search(text) for rx in DEMOTED_FORMS)


def normalize(text: str) -> str:
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    return re.sub(r"\s+", "", text).rstrip(".!?")


class EvidenceIndex:
    """근거 문장 검색(find)과 존재 확인(verify). 모델·저장소·문장 목록을 한 번만 읽는다."""

    def __init__(self, model=None, chunk_collection=None, sentence_collection=None):
        self.model = model or load_model()
        self.chunk_collection = chunk_collection or get_collection()
        self.sentence_collection = sentence_collection or get_sentence_collection()
        self._rows: dict[int, dict[str, dict]] = {}
        self._normalized: dict[int, dict[str, str]] = {}

    def rows(self, book_id: int) -> dict[str, dict]:
        if book_id not in self._rows:
            path = Paths.for_book(book_id).book_dir / SENTENCES_FILE
            if not path.exists():
                raise FileNotFoundError(f"{path} 없음 (먼저 python evidence.py 실행)")
            with path.open(encoding="utf-8") as f:
                self._rows[book_id] = {
                    r["sentence_id"]: r for r in map(json.loads, f) if r
                }
        return self._rows[book_id]

    def find(
        self, concept: str, book_id: int, exclude: frozenset[str] = frozenset()
    ) -> list[dict]:
        """개념의 근거 문장 전부를 점수순으로. exclude는 이미 쓴 sentence_id."""
        hits = search(
            concept, CHUNK_K, book_id, None,
            model=self.model, collection=self.chunk_collection,
        )  # fmt: skip
        hits = page_gate(concept, ta.filter_hits(hits, 1.0, ta.DISTANCE_MARGIN))
        if not hits:
            return []
        rows = self.rows(book_id)
        got = self.sentence_collection.get(
            where={"chunk_id": {"$in": [h["chunk_id"] for h in hits]}},
            include=["embeddings", "metadatas"],
        )
        cands = []
        for sid, vec, meta in zip(got["ids"], got["embeddings"], got["metadatas"]):
            row = rows.get(sid)
            if (
                row is None
                or row["drop_reason"]
                or meta.get("embed_hash")
                != embed_hash(row)  # 규칙이 바뀌어 남은 옛 벡터
                or sid in exclude
            ):
                continue
            cands.append((row, np.asarray(vec, dtype=np.float32), meta["section_sim"]))
        if not cands:
            return []

        query = encode(self.model, [concept])[0]
        vecs = np.stack([v for _, v, _ in cands])
        concept_sim = vecs @ query
        raw = np.maximum(concept_sim, np.array([s for _, _, s in cands]))
        penalty = np.array(
            [EXAMPLE_PENALTY if r["example_terms"] else 0.0 for r, _, _ in cands]
        )
        score = raw - penalty
        cut = statistics.median(raw.tolist()) - DELTA

        # 안내·예고형 문장은 빼지 않고 뒤로 보낸다. 점수는 그대로 두므로 포함 여부는 바뀌지 않는다
        demoted = [is_demoted_form(r["text"]) for r, _, _ in cands]
        order = sorted(range(len(cands)), key=lambda i: (demoted[i], -score[i]))
        picked: list[int] = []
        for i in order:
            if raw[i] < cut or score[i] < MIN_SCORE:
                continue
            if any(vecs[i] @ vecs[j] >= DUP_SIM for j in picked):
                continue
            picked.append(i)
        return [
            {
                **cands[i][0],
                "score": round(float(score[i]), 4),
                "concept_sim": round(float(concept_sim[i]), 4),
                "section_sim": round(float(cands[i][2]), 4),
                "demoted_form": demoted[i],
            }
            for i in picked
        ]

    def verify(self, sentence: str, book_id: int) -> str | None:
        """문장이 책에 있으면 sentence_id. 탈락 문장도 대조한다 (off의 근거 확인용).

        정규화 후 완전 일치, 다음으로 한쪽이 다른 쪽을 포함하고 길이가 80% 이상이면 일치로 본다.
        """
        target = normalize(sentence)
        if not target:
            return None
        if book_id not in self._normalized:
            self._normalized[book_id] = {
                normalize(r["text"]): sid for sid, r in self.rows(book_id).items()
            }
        table = self._normalized[book_id]
        if target in table:
            return table[target]
        for text, sid in table.items():
            short, long = sorted((target, text), key=len)
            if short in long and len(short) >= 0.8 * len(long):
                return sid
        return None


# ---------------------------------------------------------------- 실행


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="문장 단위 근거 인덱스 (bge-m3 + Chroma)"
    )
    parser.add_argument("--book-id", type=int, default=None, help="생략하면 모든 책")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="문장 저장소를 archive로 옮기고 새로 생성",
    )
    parser.add_argument(
        "--query", help="인덱스 갱신 대신 근거 문장 검색 (--book-id 필요)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.query:
        if args.book_id is None:
            parser.error("--query에는 --book-id가 필요합니다")
        results = EvidenceIndex().find(args.query, args.book_id)
        print(f"근거 문장 {len(results)}개")
        for n, r in enumerate(results, start=1):
            mark = f" 예제값{r['example_terms']}" if r["example_terms"] else ""
            mark += " (뒤로 보냄)" if r["demoted_form"] else ""
            print(f"{n:>3}. {r['score']:.3f} [{r['section']}]{mark} {r['text']}")
        return

    if args.rebuild:
        move_to_archive(SENTENCE_STORE_DIR)
    if args.book_id is not None:
        book_dirs = [Paths.for_book(args.book_id).book_dir]
    else:
        book_dirs = sorted(p.parent for p in DATA_ROOT.glob("*/chunks.jsonl"))
    model = load_model()
    collection = get_sentence_collection()
    for book_dir in book_dirs:
        if not (book_dir / "chunks.jsonl").exists():
            logger.error("chunks.jsonl 없음: %s (먼저 text_refine.py 실행)", book_dir)
            continue
        index_book(book_dir, collection, model)


if __name__ == "__main__":
    main()
