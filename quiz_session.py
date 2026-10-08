"""질문에 답하고 바로 확인 퀴즈를 낸다: 문단 검색 → 답변 → 문단마다 문제 하나씩 → 풀이.

흐름 (LLM은 llm_groq.py 또는 llm_nvidia.py. .env의 LLM_PROVIDER로 고른다):
    1. 질문과 가까운 A+ 문단을 찾는다 (text_paragraphs.ParagraphIndex).
    2. 문단에만 근거해 답하고, 답변이 인용한 문단을 고른다.
    3. 퀴즈는 모든 유형(O/X, 빈칸, 4지선다, 단답형)을 같은 방식으로 만든다: 문단 하나를 주고
       LLM이 그 문단에 근거한 문제 하나를 쓴다 (eval_blank.py의 문단 기반 생성과 같다).
       - 문단은 (가) 답변이 인용한 문단과 (나) 같은 페이지의 다른 문단을 섞어 고른다.
         (나)는 질문과 가까운 순으로 상위 몇 개 중에서 시드 고정 랜덤으로 뽑는다.
       - 프롬프트에 페이지 제목과 학습자 질문을 함께 줘서 주제에서 벗어난 문단은 LLM이 거부한다.
       - 문제의 근거 인용과 정답이 문단에 실제로 있는지 코드가 다시 검사하고(quiz_templates),
         어긋나면 다른 문단으로 다시 만든다.
    4. 문제를 낼 때마다 근거 문단과 URL을 함께 보여 준다.
퀴즈 유형별 스키마·검증·채점은 quiz_templates.py에 있다.

.env:
    LLM_PROVIDER=groq   # groq(기본) 또는 nvidia. 바꾸려면 이 값만 고친다
    GROQ_API_KEY=...
    GROQ_MODEL=...      # LLM_PROVIDER=groq일 때 필수
    NVIDIA_API_KEY=...
    NVIDIA_MODEL=...    # LLM_PROVIDER=nvidia일 때 필수 (그 밖의 선택 항목은 llm_nvidia.py 참고)

사용 예:
    python quiz_session.py "문자열 공백 제거는 어떻게 해?"
    python quiz_session.py "리스트 슬라이싱" --book-id 1 --k 4
    python quiz_session.py "딕셔너리" --quiz 6 --solve        # 퀴즈를 터미널에서 직접 풀기
    python quiz_session.py "문자열 공백 제거" --formats ox,blank --quiz 4 --seed 7
    python quiz_session.py "튜플" --no-quiz
"""

import argparse
import importlib
import json
import logging
import os
import random
import re
from collections import Counter
from dataclasses import dataclass

from dotenv import load_dotenv

import quiz_templates as qt
from text_embed import load_chunk_index, load_model
from text_paragraphs import ParagraphIndex, get_collection

# LLM 호출 파일. 둘은 같은 이름·인터페이스를 가져서 .env의 LLM_PROVIDER만 바꾸면 갈아 끼운다.
# 이 모듈을 불러오는 쪽(eval_blank.py 등)이 main보다 먼저 쓰므로 여기서 .env를 읽는다
LLM_PROVIDERS = {"groq": "llm_groq", "nvidia": "llm_nvidia"}
load_dotenv(override=True)
_provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if _provider not in LLM_PROVIDERS:
    raise SystemExit(
        f"LLM_PROVIDER={_provider!r}는 지원하지 않습니다. 사용 가능: {', '.join(LLM_PROVIDERS)}"
    )
llm = importlib.import_module(LLM_PROVIDERS[_provider])
APIError = llm.APIError
APIConnectionError = llm.APIConnectionError
APIStatusError = llm.APIStatusError
RateLimitError = llm.RateLimitError
BadRequestError = llm.BadRequestError
get_client = llm.get_client
get_model = llm.get_model

ANSWER_TEMPERATURE = 0.2
QUIZ_TEMPERATURE = 0.5
# Groq 무료 등급은 분당 출력 토큰이 1,000개라, 요청마다 상한을 명시해야 거절되지 않는다
ANSWER_MAX_TOKENS = 700
QUIZ_MAX_TOKENS = 700
# 관련 없는 문단이 근거로 섞이지 않도록 거르는 기준 (코사인 거리, 작을수록 가깝다).
# 관련 질문 15개와 무관 질문 6개로 재어 정했다: 0.45에서 관련 질문 14개가 근거를 얻고 무관 질문은
# 전부 걸러진다 (0.5는 무관 질문 3개 통과, 0.6은 5개 통과). 넓은 표현의 질문("변수란 무엇인가",
# 최고 거리 0.51)은 놓칠 수 있다
MAX_DISTANCE = 0.45  # 이보다 먼 문단은 버린다
DISTANCE_MARGIN = 0.15  # 1위보다 이만큼 이상 먼 문단은 버린다

# 퀴즈 문단 고르기
ORIGIN_CITED, ORIGIN_PAGE = "인용한 문단", "같은 페이지의 다른 문단"
PAGE_POOL_TOP = 5  # (나) 질문과 가까운 상위 몇 개 중에서 랜덤으로 뽑는다
PAGE_POOL_MIN_CHARS = 120  # (나) 너무 짧은 문단은 문제를 만들 재료가 못 된다
SLOT_TRIES = 3  # 문제 하나를 위해 문단을 바꿔 가며 시도하는 최대 횟수
CALL_RETRIES = 1  # JSON 생성 오류(json_validate_failed)일 때 같은 문단으로 다시 시도
MAX_FORMAT_CANDS = 10  # 유형을 한 번에 고르는 문단 수 상한 (넘는 문단은 허용 유형을 돌아가며 쓴다)
FORMAT_CHOICE_CHARS = 400  # 유형 고르기 프롬프트에 넣는 문단 앞부분 글자 수
FORMAT_CHOICE_MAX_TOKENS = 300
FORMAT_CHOICE_GUIDE = {
    "ox": "참/거짓이 분명한 규칙·성질 (거짓 진술은 맞는 문장의 한 곳만 바꿔 만든다)",
    "blank": "용어·명칭을 정의하는 문장 (핵심 용어 하나를 가릴 수 있다)",
    "mcq": "비슷한 개념과의 구분, 조건별 차이, 결과 비교 (그럴듯한 오답을 만들 수 있다)",
    "short": "함수·문법의 이름이나 짧은 표현 하나가 정답인 내용",
}
FORMAT_CHOICE_SYSTEM = """너는 퀴즈 출제자야. 아래 [번호] 문단마다 문제로 만들기에 가장 어울리는 유형을 하나씩 골라.
- 문단의 내용이 그 유형의 문제로 자연스럽게 만들어져야 해.
- 퀴즈 전체에서 한 유형으로 쏠리지 않게 문단 사이에 유형을 섞어.
유형:
{formats}
문단 번호 순서대로 {"formats": [{"no": 1, "format": "유형"}, ...]} 형태의 JSON 객체 하나만 출력해."""

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
CITATION = re.compile(r"\[(\d+)\]")

logger = logging.getLogger("quiz_session")

ANSWER_SYSTEM = """너는 프로그래밍 학습을 돕는 튜터야.
- 반드시 아래 [자료]에 있는 내용만 근거로 한국어로 답해.
- 문장 끝에 근거가 된 자료 번호를 [1], [2]처럼 붙여.
- 자료에 답이 없으면 추측하지 말고 "제공된 자료에서 찾을 수 없어요."라고만 답해.
- 필요하면 자료의 코드 예시를 짧게 인용해."""

QUIZ_SYSTEM = f"""너는 학습 직후 기억을 확인하는 퀴즈 출제자야.
[문단]에 근거해서 문제 1개를 만들어. 학습자는 [페이지 제목]의 내용을 공부하는 중이야.
- 문제 내용과 정답은 반드시 [문단]에 있는 내용이어야 하고, 문단 밖 지식은 쓰지 마.
- 문제는 [페이지 제목]의 주제와 [학습자 질문]에 관련된 개념이어야 해. 문단이 그 주제와 무관한 곁가지면 거부해.
- 문단 안에서 정의되지 않은 변수·값 이름은 문제에 쓰지 마. 접속어로 시작하지 마.
- evidence: 문제의 근거가 되는 구절을 [문단]에서 그대로 복사해 (한 구절, 최대 80자). 바꿔 쓰지 마.
- concept: 묻는 대상을 짧게 (예: "lstrip", "문자열 슬라이싱").
- skill은 다음 중 하나:
  "개념 이해": 동작 원리나 차이를 이해했는지
  "결과 예측": 코드 실행 결과를 맞히는지
  "함수 선택": 상황에 맞는 함수·문법을 고르는지
  "오류 찾기": 틀린 설명, 잘못된 코드, 오류가 나는 경우를 찾는지
- 문자열 값은 'hi '처럼 따옴표로 감싸서 앞뒤 공백이 보이게 써.
- 문제에 코드가 필요하면 code에 >>> 형식으로 쓰고, 필요 없으면 빈 문자열로 둬.
- 해설은 짧게 핵심만. [이미 낸 문제]가 있으면 그것과 다른 내용을 물어.
유형 규칙:
{{format}}
거부 사유 (문제를 낼 수 없을 때 reject_reason에 하나 골라):
- 정보 없음: 문단이 도입·예고·안내·잡담·감정·책 이야기뿐
- 비유: 문단 전체가 일상 사물에 빗댄 설명
- 다른 언어: 파이썬이 아닌 언어 이야기
- 추상적 주장: 확인할 수 있는 사실이 없음
- 개념 불일치: 문단이 [페이지 제목]·[학습자 질문]의 주제와 무관함
출제하면 reject_reason은 "{qt.OK_REASON}", 거부하면 문자열 필드는 빈 문자열, 배열은 빈 배열, answer는 false나 0으로 채워.
JSON 객체 하나만 출력해."""


@dataclass
class Source:
    """LLM에 넘기는 근거 하나(A+ 문단). no는 프롬프트 안의 [n] 번호."""

    no: int
    unit_id: str
    chunk_id: str
    page_id: int
    book_id: int | None
    page_title: str
    path: str  # 페이지 경로 > 소제목
    url: str
    distance: float
    text: str
    block_start: int  # 조각 안에서 이 문단이 차지하는 블록 범위
    block_end: int


class QuizFormatError(Exception):
    """모델 출력이 퀴즈 JSON 형식이 아니다."""


# ---------------------------------------------------------------- 근거 조립


def filter_hits(hits: list[dict], max_distance: float, margin: float) -> list[dict]:
    """거리순으로 정렬된 문단에서 너무 먼 것을 뺀다."""
    if not hits:
        return []
    cutoff = min(max_distance, hits[0]["distance"] + margin)
    kept = [h for h in hits if h["distance"] <= cutoff]
    dropped = [
        f"{h['unit_id']}({h['distance']:.3f})" for h in hits[len(kept) : len(kept) + 5]
    ]
    if dropped:
        logger.info("거리 기준 %.3f 초과로 제외: %s ...", cutoff, ", ".join(dropped))
    return kept


def section_path(unit: dict) -> str:
    """ "페이지 경로 > 소제목". 소제목이 페이지 제목을 대신한 문단은 경로만 쓴다."""
    path = " > ".join(unit["path"])
    return path if unit["fallback"] else f"{path} > {unit['section']}"


def page_title_of(unit: dict) -> str:
    return unit["path"][-1] if unit["path"] else ""


def retrieve(
    question: str,
    k: int,
    book_id: int | None,
    index: ParagraphIndex,
    max_distance: float = MAX_DISTANCE,
    margin: float = DISTANCE_MARGIN,
) -> list[Source]:
    hits = filter_hits(index.find(question, book_id), max_distance, margin)[:k]
    return [
        Source(
            no=no,
            unit_id=h["unit_id"],
            chunk_id=h["chunk_id"],
            page_id=h["page_id"],
            book_id=h["book_id"],
            page_title=page_title_of(h),
            path=section_path(h),
            url=h["url"],
            distance=h["distance"],
            text=h["text"],
            block_start=h["block_start"],
            block_end=h["block_end"],
        )
        for no, h in enumerate(hits, start=1)
    ]


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.no}] {s.path}\n{s.text}" for s in sources)


# ---------------------------------------------------------------- LLM


def chat(
    client,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    schema: dict | None = None,
    model: str | None = None,
    **extra,
) -> str:
    """LLM 호출은 모두 여기를 거친다. 실제 호출은 고른 provider(llm_groq / llm_nvidia)가 한다.

    model을 주지 않으면 provider의 .env 모델(GROQ_MODEL / NVIDIA_MODEL)을 쓴다 (없으면 종료).
    extra는 모델별 옵션(예: 추론 모델의 reasoning_effort)으로 그대로 넘긴다.
    """
    return llm.chat(client, messages, temperature, max_tokens, schema, model, **extra)


def answer(client, question: str, sources: list[Source]) -> str:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"[자료]\n{format_sources(sources)}\n\n[질문]\n{question}",
        },
    ]
    return chat(client, messages, ANSWER_TEMPERATURE, ANSWER_MAX_TOKENS)


def cited_sources(answer_text: str, sources: list[Source]) -> list[Source]:
    """답변이 실제로 인용한 근거만 남긴다. 인용이 없으면 빈 목록."""
    cited = {int(n) for n in CITATION.findall(answer_text)}
    return [s for s in sources if s.no in cited]


# ---------------------------------------------------------------- 퀴즈: 문단 고르기


def candidate_from_source(s: Source) -> dict:
    return {
        "unit_id": s.unit_id,
        "text": s.text,
        "url": s.url,
        "section": s.path,
        "page_title": s.page_title,
        "page_id": s.page_id,
        "book_id": s.book_id,
        "source": s.no,
        "origin": ORIGIN_CITED,
    }


def candidate_from_unit(u: dict) -> dict:
    return {
        "unit_id": u["unit_id"],
        "text": u["text"],
        "url": u["url"],
        "section": section_path(u),
        "page_title": page_title_of(u),
        "page_id": u["page_id"],
        "book_id": u["book_id"],
        "source": None,
        "origin": ORIGIN_PAGE,
    }


def same_page_pool(
    index: ParagraphIndex, question: str, cited: list[Source], rng: random.Random
) -> list[dict]:
    """(나) 인용한 문단과 같은 페이지의 다른 문단. 질문과 가까운 상위 몇 개를 섞어 앞에 둔다."""
    cited_ids = {s.unit_id for s in cited}
    pages = dict.fromkeys((s.page_id, s.book_id) for s in cited)
    found = [
        u
        for page_id, book_id in pages
        for u in index.in_page(page_id, book_id, question)
        if u["unit_id"] not in cited_ids and len(u["text"]) >= PAGE_POOL_MIN_CHARS
    ]
    found.sort(key=lambda u: -u["score"])
    top, rest = found[:PAGE_POOL_TOP], found[PAGE_POOL_TOP:]
    rng.shuffle(top)
    return [candidate_from_unit(u) for u in [*top, *rest]]


def plan_slots(total: int, pool_page_size: int, rng: random.Random) -> list[str]:
    """문제마다 문단 출처. 전체의 약 1/3을 같은 페이지의 다른 문단에서 낸다."""
    n_page = 0 if total < 2 else min(round(total / 3), pool_page_size)
    page_slots = set(rng.sample(range(total), n_page)) if n_page else set()
    return [ORIGIN_PAGE if i in page_slots else ORIGIN_CITED for i in range(total)]


# ---------------------------------------------------------------- 퀴즈: 유형 고르기


def format_choice_schema(allowed: list[str]) -> dict:
    item = {
        "type": "object",
        "properties": {
            "no": {"type": "integer"},
            "format": {"type": "string", "enum": allowed},
        },
        "required": ["no", "format"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"formats": {"type": "array", "items": item}},
        "required": ["formats"],
        "additionalProperties": False,
    }


def choose_formats(
    client, question: str, cands: list[dict], allowed: list[str]
) -> dict[str, str]:
    """문단마다 어울리는 문제 유형을 LLM이 한 번에 고른다. {unit_id: 유형}.

    허용 유형이 하나뿐이면 호출하지 않는다. 호출이 실패하거나 응답이 어긋난 문단은 빠지고,
    make_quiz가 허용 유형을 돌아가며 채운다.
    """
    if len(allowed) == 1 or not cands:
        return {c["unit_id"]: allowed[0] for c in cands}

    listing = "\n\n".join(
        f"[{n}] {c['text'][:FORMAT_CHOICE_CHARS]}" for n, c in enumerate(cands, 1)
    )
    labels = "\n".join(f"- {f}: {FORMAT_CHOICE_GUIDE[f]}" for f in allowed)
    messages = [
        {"role": "system", "content": FORMAT_CHOICE_SYSTEM.replace("{formats}", labels)},
        {"role": "user", "content": f"[학습자 질문]\n{question}\n\n{listing}"},
    ]
    try:
        text = chat(
            client,
            messages,
            ANSWER_TEMPERATURE,
            FORMAT_CHOICE_MAX_TOKENS,
            format_choice_schema(allowed),
        )
        match = JSON_OBJECT.search(text)
        picks = json.loads(match.group())["formats"] if match else []
    except (BadRequestError, json.JSONDecodeError, KeyError) as e:
        logger.warning("유형 고르기 실패, 허용 유형을 돌아가며 씁니다: %s", e)
        return {}
    chosen = {}
    for p in picks:
        no, fmt = p.get("no"), p.get("format")
        if isinstance(no, int) and 1 <= no <= len(cands) and fmt in allowed:
            chosen[cands[no - 1]["unit_id"]] = fmt
    logger.info("LLM이 고른 유형: %s", list(chosen.values()))
    return chosen


def pick_format(
    cand: dict,
    chosen: dict[str, str],
    allowed: list[str],
    done: dict[str, set[str]],
    n_made: int,
) -> str:
    """이 문단의 유형. LLM이 고른 것을 쓰되, 같은 문단에서 이미 낸 유형이면 다른 허용 유형으로 바꾼다."""
    uid = cand["unit_id"]
    fmt = chosen.get(uid) or allowed[n_made % len(allowed)]
    if fmt in done.get(uid, set()):
        fmt = next((f for f in allowed if f not in done[uid]), fmt)
    return fmt


# ---------------------------------------------------------------- 퀴즈: 문제 만들기


def failed_generation(error, limit: int = 400) -> str:
    """서버가 스키마 검사에서 거절한 모델 출력의 앞부분(Groq의 failed_generation). 실패 원인 파악용."""
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    text = str(detail.get("failed_generation", "")) if isinstance(detail, dict) else ""
    text = text.replace("\n", "⏎")
    return (text[:limit] + "…") if len(text) > limit else (text or "(없음)")


def quiz_messages(
    fmt: str, question: str, cand: dict, previous: list[str]
) -> list[dict]:
    user = (
        f"[페이지 제목]\n{cand['page_title']}\n\n[학습자 질문]\n{question}\n\n"
        f"[문단]\n{cand['text']}"
    )
    if previous:
        user += "\n\n[이미 낸 문제]\n" + "\n".join(f"- {p}" for p in previous)
    user += f"\n\n위 문단에 근거한 {qt.FORMAT_LABELS[fmt]} 문제 1개를 만들어."
    return [
        {
            "role": "system",
            "content": QUIZ_SYSTEM.replace("{format}", qt.format_instruction(fmt)),
        },
        {"role": "user", "content": user},
    ]


def ask_quiz(client, fmt: str, messages: list[dict]) -> dict:
    """LLM에 문제 하나를 요청해 JSON으로 읽는다. 형식이 어긋나면 QuizFormatError."""
    schema = qt.build_schema(fmt)
    for attempt in range(CALL_RETRIES + 1):
        try:
            text = chat(client, messages, QUIZ_TEMPERATURE, QUIZ_MAX_TOKENS, schema)
        except BadRequestError as e:
            # 모델 출력이 스키마 검사에 걸린 경우만 재시도한다. 그 밖의 400은 그대로 올린다
            if not llm.is_schema_error(e):
                raise
            logger.warning(
                "퀴즈 스키마 검사 실패 (%d회차). 거절된 출력: %s",
                attempt + 1,
                failed_generation(e),
            )
            continue
        match = JSON_OBJECT.search(text)
        if not match:
            raise QuizFormatError("JSON 객체를 찾지 못함")
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise QuizFormatError(f"JSON 파싱 실패: {e}") from e
    raise QuizFormatError("JSON 생성이 계속 실패함")


def try_question(
    client, fmt: str, question: str, cand: dict, previous: list[str]
) -> tuple[dict | None, str]:
    """문단 하나로 문제를 만든다. (문제, "") 또는 (None, 실패 사유)."""
    try:
        q = ask_quiz(client, fmt, quiz_messages(fmt, question, cand, previous))
    except QuizFormatError as e:
        return None, f"형식 오류({e})"
    if qt.is_rejected(q):
        return None, f"거부({q['reject_reason']})"
    # 실패 원인을 볼 수 있게 어떤 문제였는지 앞부분을 사유에 붙인다
    preview = " ".join(str(q.get("statement") or q.get("question") or "").split())[:60]
    if problem := qt.validate(fmt, q):
        return None, f"형식 결함({problem}) → {preview}"
    if problem := qt.check_against(fmt, q, cand["text"]):
        return None, f"검증 실패({problem}) → {preview}"
    q = qt.normalize(fmt, q)
    q["paragraph"] = {
        k: cand[k]
        for k in ("unit_id", "url", "section", "text", "page_title", "origin")
    }
    q["source"] = cand["source"]
    return q, ""


def make_quiz(
    client,
    question: str,
    cited: list[Source],
    index: ParagraphIndex,
    total: int,
    allowed: list[str],
    rng: random.Random | None = None,
) -> list[dict]:
    """total개 문제를 만든다. 문제 유형은 allowed 안에서 LLM이 문단마다 고른다.

    문제마다 문단 하나에서 LLM 호출 한 번이 기본이고, 유형 고르기에 퀴즈 전체로 한 번이 더 든다.
    거부되거나 검증에 실패하면 같은 출처의 다음 문단(없으면 다른 출처)으로 최대 SLOT_TRIES번 시도한다.
    """
    rng = rng or random.Random()
    pools = {
        ORIGIN_CITED: [candidate_from_source(s) for s in cited],
        ORIGIN_PAGE: same_page_pool(index, question, cited, rng),
    }
    slots = plan_slots(total, len(pools[ORIGIN_PAGE]), rng)
    cands = [*pools[ORIGIN_CITED], *pools[ORIGIN_PAGE]][:MAX_FORMAT_CANDS]
    chosen = choose_formats(client, question, cands, allowed)
    used: Counter[str] = Counter()  # 문단을 골고루 쓰려고 쓴 횟수가 적은 것부터 고른다
    done: dict[str, set[str]] = {}  # 문단마다 이미 낸 유형
    rejected: set[str] = set()  # LLM이 거부한 문단
    quiz: list[dict] = []
    previous: list[str] = []

    for origin in slots:
        other = ORIGIN_PAGE if origin == ORIGIN_CITED else ORIGIN_CITED
        order = [
            c
            for c in (
                *sorted(pools[origin], key=lambda c: used[c["unit_id"]]),
                *sorted(pools[other], key=lambda c: used[c["unit_id"]]),
            )
            if c["unit_id"] not in rejected  # LLM이 문단 자체를 거부했으면 다시 시키지 않는다
        ]
        for cand in order[:SLOT_TRIES]:
            fmt = pick_format(cand, chosen, allowed, done, len(quiz))
            q, reason = try_question(client, fmt, question, cand, previous)
            if q is None:
                logger.info("[%s] %s 문단 실패: %s", fmt, cand["origin"], reason)
                if reason.startswith("거부("):
                    rejected.add(cand["unit_id"])
                continue
            used[cand["unit_id"]] += 1
            done.setdefault(cand["unit_id"], set()).add(fmt)
            previous.append(q.get("statement") or q["question"])
            quiz.append(q)
            break
        else:
            logger.warning("문제를 만들지 못함 (%d번 시도)", SLOT_TRIES)

    for q in quiz:
        if q["format"] == "mcq":
            qt.shuffle_mcq(q, rng)
    if len(quiz) < total:
        logger.warning("요청 %d문제 중 %d문제만 만들었습니다", total, len(quiz))
    logger.info(
        "퀴즈 %d문제: %s",
        len(quiz),
        [f"{q['format']}/{q['paragraph']['origin']}/{q['concept']}" for q in quiz],
    )
    return quiz


# ---------------------------------------------------------------- 출력·풀이


def print_evidence(q: dict) -> None:
    """문제의 근거 문단과 URL. 문제를 낼 때 어느 문단에서 나왔는지 보여 준다."""
    p = q["paragraph"]
    print(f"   근거({p['origin']}): {p['section']}")
    print(f"   {p['url']}")
    print(qt.indent(p["text"], "   │ "))


def print_quiz(questions: list[dict], solve: bool) -> None:
    auto_total = auto_correct = 0
    for i, q in enumerate(questions, start=1):
        print(f"\nQ{i}. {qt.render_question(q)}")

        if solve:
            reply = input("   답: ").strip()
            # 메타인지 점검: 정답을 보기 전에 스스로 확신도를 매긴다
            confidence = input("   확신도 (1 모름 / 2 애매 / 3 확실): ").strip() or "-"
            result = qt.grade(q, reply)
            if result is None:
                print(f"   → 확신도 {confidence}, 아래 정답과 비교해 보세요")
            else:
                auto_total += 1
                auto_correct += result
                print(f"   → {'정답' if result else '오답'} (확신도 {confidence})")

        print(qt.indent(qt.render_answer(q)))
        print_evidence(q)

    if solve and auto_total:
        print(f"\n자동 채점 {auto_total}문제 중 {auto_correct}문제 정답")


def parse_formats(value: str) -> list[str]:
    formats = list(dict.fromkeys(f.strip() for f in value.split(",") if f.strip()))
    unknown = [f for f in formats if f not in qt.FORMATS]
    if unknown or not formats:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 유형 {unknown}. 사용 가능: {','.join(qt.FORMATS)}"
        )
    return formats


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description=f"RAG 기반 답변 + 확인 퀴즈 ({llm.NAME})")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5, help="근거 후보 문단 수 (최대)")
    parser.add_argument("--book-id", type=int, default=None, help="검색할 책 제한")
    parser.add_argument(
        "--max-distance", type=float, default=MAX_DISTANCE, help="근거 거리 상한"
    )
    parser.add_argument(
        "--margin", type=float, default=DISTANCE_MARGIN, help="1위 대비 허용 거리 차"
    )
    parser.add_argument(
        "--formats",
        type=parse_formats,
        default=None,
        help=f"허용할 퀴즈 유형 (쉼표로 구분, 사용 가능: {','.join(qt.FORMATS)}). "
        "생략하면 전부 허용하고 LLM이 문단마다 고른다. 하나만 주면 그 유형으로 고정",
    )
    parser.add_argument("--quiz", type=int, default=4, help="총 문제 수")
    parser.add_argument(
        "--seed", type=int, default=None, help="문단 고르기·보기 섞기 시드 (재현용)"
    )
    parser.add_argument("--no-quiz", action="store_true")
    parser.add_argument("--solve", action="store_true", help="퀴즈를 직접 풀기")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    get_model()  # 모델 설정이 없으면 임베딩을 불러오기 전에 바로 종료
    client = get_client()
    index = ParagraphIndex(load_model(), get_collection(), load_chunk_index())
    sources = retrieve(
        args.question,
        args.k,
        args.book_id,
        index,
        args.max_distance,
        args.margin,
    )
    if not sources:
        # LLM을 부르지 않고 끝낸다 (임베딩이 없거나, 관련 자료가 거리 기준 밖)
        print("관련 자료를 찾지 못했어요. 거리 기준(--max-distance)도 확인해 보세요.")
        return

    try:
        print("\n=== 답변 ===")
        answer_text = answer(client, args.question, sources)
        print(answer_text)
        print("\n=== 출처 ===")
        for s in sources:
            print(f"[{s.no}] ({s.distance:.3f}) {s.path}\n    {s.url}")

        if not args.no_quiz:
            quiz_sources = cited_sources(answer_text, sources)
            if not quiz_sources:
                print("\n답변에 인용된 자료가 없어 퀴즈를 건너뜁니다.")
                return
            print("\n=== 확인 퀴즈 ===")
            questions = make_quiz(
                client,
                args.question,
                quiz_sources,
                index,
                args.quiz,
                args.formats or list(qt.FORMATS),
                random.Random(args.seed),
            )
            print_quiz(questions, args.solve)
    except APIConnectionError as e:
        logger.error("%s 연결 실패: %s", llm.NAME, e)
    except APIStatusError as e:
        logger.error("%s API 오류 (%s): %s", llm.NAME, e.status_code, e.message)
    except QuizFormatError as e:
        logger.error("%s", e)


if __name__ == "__main__":
    main()
