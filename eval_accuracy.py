"""정확도 테스트: RAG 없이/있이 O/X 문제를 만들고, 책에서 근거 문장을 찾아 붙인다.

기획서 테스트케이스 2 "정확도 판단 — 출력한 문제의 개념이 RAG 문서에 있는가?"를
사람이 판정하기 쉽게 돕고, 자동 지표로 보조하기 위한 실험 도구다.

케이스마다 (책, 개념)으로 off/on 각각 한 문제씩 남긴다.
    off: 책 이름과 개념만 주고 O/X 문제 생성 (비교군)
    on : 개념으로 그 책을 검색한 근거 조각을 함께 주고 생성 (실험군)
    정답(O/X)은 케이스마다 무작위로 정해 지시한다. 모델이 X로 쏠리는 것을 막고,
    같은 케이스의 off와 on은 같은 지정 정답을 써서 짝 비교가 공정하게 한다.
하나 빼기: off/on마다 후보를 --candidates개(기본 2) 연달아 만들고, 근거_sim이 가장
    높은 문제만 저장한다. 뒤 후보에는 앞 후보를 넘겨 다른 내용을 묻게 한다
    (같은 문제 둘 중에서 고르면 의미가 없다). 탈락한 후보는 로그에만 남긴다.
근거 문장: on의 참고 자료 조각 + 문제로 책 전체를 검색한 상위 조각에서,
    문제와 가장 비슷한 문장 하나와 그 위치(URL, 소제목 앵커 포함)를 찾는다.
    참고 자료 밖(같은 책의 다른 페이지)에 근거가 있는 경우도 잡기 위해 책 검색을 더한다.
지표:
    근거_sim : 문제 ↔ 근거 문장의 bge-m3 코사인 유사도 (1에 가까울수록 비슷).
               문장 단위라 조각 단위보다 근거 유무를 잘 가른다
    근거 위치: 근거 문장이 on의 참고 자료 안에 있는지 밖(같은 책의 다른 곳)에 있는지.
               on이 "자료에 있는 내용만" 지시를 지켰는지 보는 데 쓴다
    (조각 벡터 기반 ref_sim·book_sim은 긴 페이지에서 근거가 희석돼 판정을 오도하고,
     책 검색 결과는 이미 근거 문장 후보에 포함돼 있어 뺐다)

판정 열 (사람이 채운다):
    근거 확인(o/x): 학습한 책 어딘가에 문제의 참/거짓을 판정할 근거가 있으면 o
    정답 확인(o/x): 문제가 명확하고 O/X 정답이 맞으면 o

조각 ID는 구글 시트가 "6-01"을 날짜로 바꾸지 않도록 "c6-01"처럼 c를 붙여 쓴다.
결과는 .data/eval/accuracy_YYYYMMDD_HHMMSS_모델.csv에 한 줄씩 바로 쓴다 (중간에 끊겨도 남는다).
Groq 하루 토큰 한도(TPD)에 걸리면 남은 케이스를 건너뛰고 그때까지의 결과만 저장한다.
--resume으로 이전 결과 파일을 주면 성공한 (케이스, rag)는 그대로 가져오고 나머지만
실행해 새 파일에 합친다. 지정 답은 케이스 순서로 정해지므로 --ids·--seed를 같게 줘야 한다.

사용 예:
    python eval_accuracy.py --dry-run          # LLM 없이 책 매핑·검색만 확인
    python eval_accuracy.py --ids 1 10 16      # 일부 케이스만 (Groq 12회 호출)
    python eval_accuracy.py --candidates 1     # 후보 없이 1문제씩 (하나 빼기 끔)
    python eval_accuracy.py                    # 30개 전체 (Groq 120회 호출)
    python eval_accuracy.py --model qwen/qwen3.8-27b
    python eval_accuracy.py --resume .data/eval/accuracy_20261006_1536_gpt-oss-20b.csv
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError

import text_answer as ta
from text_embed import (
    DATA_ROOT,
    IMAGE_PLACEHOLDER,
    embed,
    get_collection,
    load_model,
    search,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CASES_CSV = (
    PROJECT_ROOT / ".idea_folder" / "testcase" / "테스트 케이스.정확도 - 시트1.csv"
)
OUT_DIR = DATA_ROOT / "eval"
# qwen3.8-27b는 하루 토큰 한도(20만)를 전체 실행 한 번에 거의 다 써서, 빠른 테스트용으로
# 바꿨다. 모델이 바뀌면 이전 결과와 직접 비교할 수 없으니 파일 이름에 모델명을 넣는다
DEFAULT_EVAL_MODEL = "openai/gpt-oss-20b"
# O/X 1문제는 짧다. 분당 출력 토큰 한도(1,000)를 아끼려고 작게 잡는다
OX_MAX_TOKENS = 200
# 추론 모델은 생각 토큰도 max_tokens에 포함돼 200이면 JSON 전에 잘린다.
# 추론 강도를 낮추고 상한을 넉넉히 잡는다 (Groq 추론 모델 옵션)
MODEL_OPTIONS = {
    "openai/gpt-oss": {"max_tokens": 800, "reasoning_effort": "low"},
}
TEMPERATURE = 0.3
GROQ_MAX_RETRIES = 6  # 분당 한도에 걸리면 SDK가 대기 후 재시도한다
EVIDENCE_BOOK_K = 3  # 근거 문장 후보에 더할 책 전체 검색 상위 조각 수
MIN_SENTENCE_CHARS = 8
SENTENCE_BATCH_SIZE = 32
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
TEXT_BLOCKS = ("paragraph", "note", "list", "key_point", "caption", "subheading")

OX_SCHEMA = {
    "type": "object",
    "properties": {"statement": {"type": "string"}, "answer": {"type": "boolean"}},
    "required": ["statement", "answer"],
    "additionalProperties": False,
}
COMMON_RULES = """- O/X 문제 1개를 한국어로 만들어.
- statement는 참인지 거짓인지 판단할 진술문 한 문장, answer는 참이면 true, 거짓이면 false.
- 정답은 요청에서 지정한 대로 맞춰. O면 맞는 진술, X면 맞는 문장에서 한 곳만 바꾼 거짓 진술.
- JSON 객체 하나만 출력해."""
DEFAULT_CANDIDATES = 2
DEFAULT_SEED = 42
# 지정 정답 배정: O/X 개수가 같으면 반반, 한쪽이 많으면 적은 쪽을 이 확률로 뽑는다.
# 독립 추첨은 30케이스에서 21:9 이상 치우칠 확률이 약 4%이고 시드가 고정이라 매번
# 같은 쏠림이 반복된다. 이 방식은 시뮬레이션에서 17:13 이내가 97%다 (Efron의 biased coin)
BALANCE_P = 2 / 3
# on이 off보다 근거_sim이 낮아도 이 범위 안이면 신호로 보지 않는다 (오차 범위).
# 1차 결과(gpt-oss-20b, 30케이스)의 음수 차이가 0.002~0.021과 0.047~0.076으로
# 뚜렷이 나뉘었고, 거의 같은 뜻의 문장끼리도 이 정도 유사도 차이는 흔하다
SIGNAL_MARGIN = 0.03
OFF_SYSTEM = f"""너는 파이썬 학습 퀴즈 출제자야.
학습자가 아래 책에서 아래 개념을 공부했어. 이 개념을 확인하는 문제를 만들어.
{COMMON_RULES}"""
ON_SYSTEM = f"""너는 파이썬 학습 퀴즈 출제자야.
학습자가 아래 책에서 아래 개념을 공부했어. 이 개념을 확인하는 문제를 만들어.
- 반드시 [자료]에 있는 내용만 근거로 만들어. 자료 밖 지식은 쓰지 마.
{COMMON_RULES}"""

# 행에는 있지만 CSV에 쓰지 않는 값: 학습한 자료(book_id로 충분), 지정 답(요약 집계용)
FIELDS = [
    "id",
    "book_id",
    "rag",
    "개념",
    "문제",
    "답",
    "근거 문장",
    "근거 URL",
    "근거_sim",
    "근거 위치",  # 참고 자료 안 / 참고 자료 밖
    "근거 확인(o/x)",  # o: 학습한 책에 문제의 참/거짓을 판정할 근거가 있음
    "정답 확인(o/x)",  # o: 문제가 명확하고 O/X 정답이 맞음
    "참고 자료",
    "근거 조각",
    "오류",
]

logger = logging.getLogger("eval_accuracy")


def label(chunk_id: str) -> str:
    """시트에서 날짜로 바뀌지 않게 접두어를 붙인 조각 ID."""
    return f"c{chunk_id}"


# ---------------------------------------------------------------- 준비


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("id", "").strip()]


def book_ids_by_title(index: dict[str, dict]) -> dict[str, int]:
    """chunks의 book_title → book_id. CSV의 책 이름은 판차 표기가 빠져 있어 부분 일치로 찾는다."""
    titles = {}
    for chunk in index.values():
        if chunk.get("book_title") and chunk.get("book_id") is not None:
            titles[chunk["book_title"]] = chunk["book_id"]
    return titles


def resolve_book(name: str, titles: dict[str, int]) -> int | None:
    matches = {bid for title, bid in titles.items() if name.strip() in title}
    return matches.pop() if len(matches) == 1 else None


# ---------------------------------------------------------------- 생성


class DailyLimitReached(Exception):
    """Groq 하루 한도(토큰·요청). 기다려도 바로 풀리지 않으니 실행을 멈춘다."""


def model_options(model: str) -> dict:
    for prefix, options in MODEL_OPTIONS.items():
        if model.startswith(prefix):
            return dict(options)
    return {"max_tokens": OX_MAX_TOKENS}


def generate(client, model: str, system: str, user: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    options = model_options(model)
    max_tokens = options.pop("max_tokens")
    try:
        text = ta.chat(
            client, messages, TEMPERATURE, max_tokens, OX_SCHEMA, model=model, **options
        )
    except RateLimitError as e:
        if "per day" in str(e):
            raise DailyLimitReached(str(e)) from e
        raise
    match = ta.JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"JSON 없음: {text[:120]}")
    data = json.loads(match.group())  # JSONDecodeError는 ValueError의 하위 클래스
    statement = str(data.get("statement", "")).strip()
    if not isinstance(data.get("answer"), bool) or not statement:
        raise ValueError(f"형식 오류: {text[:120]}")
    return statement, "O" if data["answer"] else "X"


def request(target: str, previous: list[str]) -> str:
    """문항별 요청: 지정 정답과, 같은 케이스에서 이미 낸 문제(반복 방지)."""
    text = f"[요청]\n정답이 {target}인 문제를 만들어."
    if previous:
        listed = "\n".join(f"- {p}" for p in previous)
        text += f"\n아래 문제와 다른 내용을 물어:\n{listed}"
    return text


def off_prompt(book: str, concept: str, target: str, previous: list[str]) -> str:
    return f"[책]\n{book}\n\n[개념]\n{concept}\n\n{request(target, previous)}"


def on_prompt(
    book: str, concept: str, sources: list, target: str, previous: list[str]
) -> str:
    return (
        f"[자료]\n{ta.format_sources(sources)}\n\n[책]\n{book}\n\n[개념]\n{concept}"
        f"\n\n{request(target, previous)}"
    )


# ---------------------------------------------------------------- 근거 문장


def _walk(blocks: list[dict]):
    for block in blocks:
        if block["type"] == "concept_box":
            yield from _walk(block["blocks"])
        else:
            yield block


def _block_lines(block: dict) -> list[str]:
    kind = block["type"]
    if kind in TEXT_BLOCKS:
        text = IMAGE_PLACEHOLDER.sub("", block.get("text", ""))
        return SENTENCE_SPLIT.split(text)
    if kind == "table":
        return [" | ".join(row) for row in block["rows"]]
    if kind == "code_example":
        # 실행 예시는 "입력 → 출력" 한 줄로 근거가 된다 (예: a.lstrip() → 'hi ')
        return [
            f"{p['input']} → {p['output']}" if p.get("output") else p["input"]
            for p in block.get("pairs") or []
        ]
    return []


def chunk_sentences(chunk: dict) -> list[tuple[str, str]]:
    """조각의 (문장, 그 문장이 속한 위치 URL) 목록. 소제목을 지나면 그 앵커로 바뀐다."""
    base = chunk["source_url"].split("#")[0]
    url = chunk["source_url"]
    result = []
    for block in _walk(chunk["blocks"]):
        if block["type"] == "heading":
            if block["level"] in (2, 3) and block.get("anchor"):
                url = f"{base}#{block['anchor']}"
            continue
        for line in _block_lines(block):
            line = line.strip()
            if len(line) >= MIN_SENTENCE_CHARS:
                result.append((line, url))
    return result


class SentenceIndex:
    """조각별 문장과 문장 임베딩을 한 번만 계산해 재사용한다 (on/off·케이스 간 공유)."""

    def __init__(self, model, index: dict[str, dict]):
        self.model = model
        self.index = index
        self.cache: dict[str, tuple[list[tuple[str, str]], np.ndarray]] = {}

    def get(self, chunk_id: str) -> tuple[list[tuple[str, str]], np.ndarray]:
        if chunk_id not in self.cache:
            sentences = chunk_sentences(self.index[chunk_id])
            # 짧은 문장이 많아 배치를 키우고, 조각마다 뜨는 진행 막대는 끈다
            vecs = (
                self.model.encode(
                    [s for s, _ in sentences],
                    batch_size=SENTENCE_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if sentences
                else np.empty((0, 0))
            )
            self.cache[chunk_id] = (sentences, vecs)
        return self.cache[chunk_id]

    def best(self, vec: np.ndarray, chunk_ids: list[str]) -> dict:
        best = {"근거_sim": "", "근거 문장": "", "근거 URL": "", "근거 조각": ""}
        top = -1.0
        for chunk_id in dict.fromkeys(chunk_ids):
            if chunk_id not in self.index:
                continue
            sentences, vecs = self.get(chunk_id)
            if not sentences:
                continue
            sims = vecs @ vec
            n = int(np.argmax(sims))
            if sims[n] > top:
                top = float(sims[n])
                best = {
                    "근거_sim": f"{top:.4f}",
                    "근거 문장": sentences[n][0],
                    "근거 URL": sentences[n][1],
                    "근거 조각": label(chunk_id),
                }
        return best


# ---------------------------------------------------------------- 지표

IN_REF, OUT_REF = "참고 자료 안", "참고 자료 밖"


def measure(
    model,
    collection,
    sentences: SentenceIndex,
    text: str,
    ref_ids: list[str],
    book_id: int,
) -> dict:
    """근거 문장(근거_sim, URL)과 그 문장이 참고 자료 안에 있는지."""
    vec = np.asarray(embed(model, [text])[0], dtype=np.float32)
    hits = search(
        text, EVIDENCE_BOOK_K, book_id, None, model=model, collection=collection
    )
    candidates = [*ref_ids, *(h["chunk_id"] for h in hits)]
    result = sentences.best(vec, candidates)
    if result["근거 조각"]:
        in_ref = result["근거 조각"] in {label(i) for i in ref_ids}
        result["근거 위치"] = IN_REF if in_ref else OUT_REF
    return result


# ---------------------------------------------------------------- 실행


class TargetAssigner:
    """지정 정답(O/X)을 무작위로 정하되 한쪽으로 크게 치우치지 않게 한다.

    반반을 강제하지는 않는다: 개수가 같으면 반반, 다르면 적은 쪽을 BALANCE_P로 뽑는다.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counts = {"O": 0, "X": 0}

    def next(self) -> str:
        n_o, n_x = self.counts["O"], self.counts["X"]
        p_o = 0.5 if n_o == n_x else (BALANCE_P if n_o < n_x else 1 - BALANCE_P)
        target = "O" if self.rng.random() < p_o else "X"
        self.counts[target] += 1
        return target


def pick_best(candidates: list[dict]) -> dict:
    """근거_sim이 가장 높은 후보. 모두 실패했으면 마지막 실패 기록을 남긴다."""
    done = [c for c in candidates if c.get("근거_sim")]
    if not done:
        return candidates[-1]
    return max(done, key=lambda c: float(c["근거_sim"]))


def load_finished(path: Path) -> list[dict]:
    """이전 결과에서 문제 생성에 성공한 행만. 오류 행은 다시 실행하도록 버린다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"--resume 파일의 열이 현재 형식과 다릅니다: {sorted(missing)}"
            )
        return [r for r in reader if r.get("문제") and not r.get("오류")]


def summarize(rows: list[dict], margin: float = SIGNAL_MARGIN) -> None:
    print("\n=== 근거_sim (계산된 행만) ===")
    for rag in ("off", "on"):
        values = [
            float(r["근거_sim"]) for r in rows if r["rag"] == rag and r.get("근거_sim")
        ]
        if values:
            print(
                f"  {rag:3} n={len(values):2}  평균 {statistics.mean(values):.4f}"
                f"  중앙값 {statistics.median(values):.4f}  최소 {min(values):.4f}"
            )
    # 같은 케이스(같은 지정 답)의 on − off 차이 (짝지은 비교)
    pairs: dict[str, dict] = {}
    for r in rows:
        pairs.setdefault(r["id"], {})[r["rag"]] = r
    diffs = {
        cid: float(p["on"]["근거_sim"]) - float(p["off"]["근거_sim"])
        for cid, p in pairs.items()
        if "on" in p
        and "off" in p
        and p["on"].get("근거_sim")
        and p["off"].get("근거_sim")
    }
    if diffs:
        flagged = sorted(
            (cid for cid, d in diffs.items() if d < -margin), key=lambda c: diffs[c]
        )
        within = sum(-margin <= d < 0 for d in diffs.values())
        print(f"  on−off: 평균 {statistics.mean(diffs.values()):+.4f}")
        print(
            f"  신호 (on이 off보다 {margin} 넘게 낮음): {len(flagged)}/{len(diffs)} "
            + str([f"{cid}({diffs[cid]:+.3f})" for cid in flagged])
        )
        print(f"  오차 범위 안에서 낮음 (신호 아님): {within}")
    # 실험군(on)이 참고 자료 밖에서 근거를 찾았다면 자료 밖 지식을 쓴 신호
    print("\n=== 근거 위치 ===")
    for rag in ("off", "on"):
        located = [r for r in rows if r["rag"] == rag and r.get("근거 위치")]
        if located:
            inside = sum(r["근거 위치"] == IN_REF for r in located)
            print(f"  {rag:3} {IN_REF} {inside} / {OUT_REF} {len(located) - inside}")
    # 정답 쏠림과 지정 답 준수
    print("\n=== 정답 O/X ===")
    for rag in ("off", "on"):
        done = [r for r in rows if r["rag"] == rag and r.get("답")]
        if done:
            n_o = sum(r["답"] == "O" for r in done)
            # 지정 답은 파일에 쓰지 않으므로 --resume으로 가져온 행은 불일치 집계에서 빠진다
            assigned = [r for r in done if r.get("지정 답")]
            mismatch = sum(r["답"] != r["지정 답"] for r in assigned)
            print(
                f"  {rag:3} O {n_o} / X {len(done) - n_o}, "
                f"지정 답과 다름 {mismatch}/{len(assigned)}"
            )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG on/off O/X 문제 정확도 테스트")
    parser.add_argument("--cases", type=Path, default=CASES_CSV)
    parser.add_argument("--ids", nargs="+", help="실행할 케이스 id (생략하면 전체)")
    parser.add_argument("--k", type=int, default=5, help="RAG 근거 후보 조각 수")
    # text_answer의 절대 상한(0.5)을 쓰면 book110의 짧은 개념명(리스트, 딕셔너리 등)은
    # 근거가 0개가 되어 on을 비교할 수 없다. 기본은 1위 대비 차이 기준만 쓴다
    parser.add_argument(
        "--max-distance", type=float, default=1.0, help="근거 거리 상한 (기본: 끔)"
    )
    parser.add_argument("--margin", type=float, default=ta.DISTANCE_MARGIN)
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="off/on마다 만들 후보 수. 근거_sim이 가장 높은 하나만 저장",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="지정 정답(O/X) 무작위 시드"
    )
    parser.add_argument(
        "--model", default=DEFAULT_EVAL_MODEL, help="문제 생성 LLM (Groq 모델 ID)"
    )
    parser.add_argument(
        "--resume", type=Path, help="이전 결과 CSV. 성공한 (케이스, rag)는 건너뛴다"
    )
    parser.add_argument(
        "--signal-margin",
        type=float,
        default=SIGNAL_MARGIN,
        help="on이 off보다 이만큼 넘게 낮을 때만 신호로 본다",
    )
    parser.add_argument("--dry-run", action="store_true", help="LLM 없이 검색만 확인")
    args = parser.parse_args()
    assigner = TargetAssigner(random.Random(args.seed))
    llm_model = args.model  # model은 아래에서 임베딩 모델(bge-m3) 이름으로 쓴다

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cases = load_cases(args.cases)
    if args.ids:
        cases = [c for c in cases if c["id"].strip() in set(args.ids)]
    index = ta.load_chunk_index()
    titles = book_ids_by_title(index)
    model, collection = load_model(), get_collection()
    sentences = SentenceIndex(model, index)

    client = None
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if not args.dry_run:
        client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=GROQ_MAX_RETRIES)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.resume:
            # 새 파일을 열기 전에 먼저 읽는다. 이전 파일은 그대로 두고 성공한 행만 옮긴다
            rows = load_finished(args.resume)
            done = {(r["id"], r["rag"]) for r in rows}
            logger.info(
                "이어서 실행: 성공한 %d행을 가져옴 (%s)", len(rows), args.resume
            )
        slug = llm_model.split("/")[-1]
        stamp = f"{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        out = OUT_DIR / f"accuracy_{stamp}_{slug}.csv"
        if args.resume and out.resolve() == args.resume.resolve():
            raise SystemExit(f"결과 파일이 --resume 파일과 같습니다: {out}")
        writer_file = out.open("w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(writer_file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer_file.flush()

    stop_reason: DailyLimitReached | None = None
    for case in cases:
        cid, book, concept = case["id"].strip(), case["학습한 자료"], case["개념"]
        book_id = resolve_book(book, titles)
        if book_id is None:
            logger.error("[%s] 책을 찾지 못함: %s (후보: %s)", cid, book, list(titles))
            continue
        # off와 on이 같은 지정 정답을 쓰도록 케이스마다 한 번만 정한다.
        # 건너뛰는 케이스에서도 뽑아야 --resume 때 같은 케이스가 같은 지정 답을 받는다
        target = assigner.next()
        if (cid, "off") in done and (cid, "on") in done:
            continue
        sources = ta.retrieve(
            concept,
            args.k,
            book_id,
            model,
            collection,
            index,
            args.max_distance,
            args.margin,
        )
        ref_ids = [s.chunk_id for s in sources]
        logger.info(
            "[%s] %s / %s → book %s, 근거 %s",
            cid, book, concept, book_id,
            [f"{s.chunk_id}({1 - s.distance:.3f})" for s in sources],
        )  # fmt: skip
        if args.dry_run:
            continue

        for rag in ("off", "on"):
            if (cid, rag) in done:
                continue
            base = {
                "id": cid,
                "학습한 자료": book,
                "개념": concept,
                "book_id": book_id,
                "rag": rag,
                "지정 답": target,
                "참고 자료": " ".join(label(i) for i in ref_ids),
            }
            if rag == "on" and not sources:
                best = {**base, "오류": "근거 없음 (거리 기준 밖)"}
            else:
                system = OFF_SYSTEM if rag == "off" else ON_SYSTEM
                previous: list[str] = []  # 앞 후보. 뒤 후보가 다른 내용을 묻게 한다
                candidates = []
                for n in range(1, args.candidates + 1):
                    user = (
                        off_prompt(book, concept, target, previous)
                        if rag == "off"
                        else on_prompt(book, concept, sources, target, previous)
                    )
                    row = {**base}
                    tag = f"{cid}/{rag}/후보{n}"
                    try:
                        row["문제"], row["답"] = generate(
                            client, llm_model, system, user
                        )
                        previous.append(row["문제"])
                        if row["답"] != target:
                            logger.warning(
                                "[%s] 지정 답 %s와 다른 답 %s", tag, target, row["답"]
                            )
                        row.update(
                            measure(
                                model,
                                collection,
                                sentences,
                                row["문제"],
                                ref_ids,
                                book_id,
                            )
                        )
                        logger.info(
                            "[%s] %s (%s) 근거 %s %s",
                            tag, row["문제"], row["답"], row["근거_sim"], row.get("근거 위치", ""),
                        )  # fmt: skip
                    except DailyLimitReached as e:
                        stop_reason = e
                        break
                    except ValueError as e:
                        row["오류"] = str(e)
                        logger.warning("[%s] 실패: %s", tag, e)
                    except APIError as e:  # 재시도 후에도 실패한 Groq 오류는 기록
                        row["오류"] = f"{type(e).__name__}: {e}"
                        logger.warning("[%s] API 실패: %s", tag, row["오류"])
                    candidates.append(row)
                if stop_reason:
                    break  # 이 (케이스, rag)는 저장하지 않는다 → --resume 때 다시 만든다
                best = pick_best(candidates)
                for loser in candidates:
                    if loser is not best and loser.get("문제"):
                        logger.info(
                            "[%s/%s] 탈락 (근거 %s): %s",
                            cid, rag, loser.get("근거_sim"), loser["문제"],
                        )  # fmt: skip
            rows.append(best)
            writer.writerow(best)
            writer_file.flush()
        if stop_reason:
            break

    if not args.dry_run:
        writer_file.close()
        if stop_reason:
            logger.error("하루 한도에 걸려 중단했습니다: %s", stop_reason)
            print(
                f"\n하루 한도로 중단. 한도가 풀리면 이어서 실행하세요:\n"
                f"  python eval_accuracy.py --resume {out} --model {llm_model}"
                + (f" --ids {' '.join(args.ids)}" if args.ids else "")
            )
        summarize(rows, args.signal_margin)
        print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
