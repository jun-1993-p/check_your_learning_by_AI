"""질문에 답하고 바로 확인 퀴즈를 낸다: 문단 검색 → 답변 → 문단마다 문제 하나씩 → 풀이.

흐름 (Groq LLM):
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
    GROQ_API_KEY=...
    GROQ_MODEL=qwen/qwen3.8-27b   # 선택. 생략하면 기본값

사용 예:
    python quiz_session.py "문자열 공백 제거는 어떻게 해?"
    python quiz_session.py "리스트 슬라이싱" --book-id 1 --k 4
    python quiz_session.py "딕셔너리" --quiz 6 --solve        # 퀴즈를 터미널에서 직접 풀기
    python quiz_session.py "문자열 공백 제거" --formats ox,blank --quiz 4 --seed 7
    python quiz_session.py "튜플" --no-quiz
"""

import argparse
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

DEFAULT_MODEL = "qwen/qwen3.8-27b"
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

THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)
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


def get_client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(".env에 GROQ_API_KEY가 없습니다")
    return Groq(api_key=api_key)


def chat(
    client,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    schema: dict | None = None,
    model: str | None = None,
    **extra,
) -> str:
    """LLM 호출은 모두 여기를 거친다. 로컬 모델로 바꿀 때 이 함수만 교체하면 된다.

    model을 주지 않으면 .env의 GROQ_MODEL(없으면 기본값)을 쓴다.
    extra는 모델별 옵션(예: 추론 모델의 reasoning_effort)으로 그대로 넘긴다.
    """
    options = dict(extra)
    if schema is not None:
        # strict가 아니면 Groq는 생성을 제한하지 않고 생성 후 검사만 해서,
        # 모델이 스키마를 무시하면 400(json_validate_failed)으로 통째로 거절된다
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "quiz", "schema": schema, "strict": True},
        }
    response = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **options,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning("출력이 max_tokens(%d)에서 잘렸습니다", max_tokens)
    # 추론 모델이 생각 과정을 본문에 섞어 내보내는 경우를 걸러낸다
    return THINK_TAG.sub("", choice.message.content or "").strip()


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


def plan_slots(
    counts: dict[str, int], pool_page_size: int, rng: random.Random
) -> list[tuple[str, str]]:
    """문제마다 (유형, 문단 출처). 전체의 약 1/3을 같은 페이지의 다른 문단에서 낸다."""
    formats = [fmt for fmt, n in counts.items() for _ in range(n)]
    total = len(formats)
    n_page = 0 if total < 2 else min(round(total / 3), pool_page_size)
    page_slots = set(rng.sample(range(total), n_page)) if n_page else set()
    return [
        (fmt, ORIGIN_PAGE if i in page_slots else ORIGIN_CITED)
        for i, fmt in enumerate(formats)
    ]


# ---------------------------------------------------------------- 퀴즈: 문제 만들기


def failed_generation(error, limit: int = 400) -> str:
    """Groq가 스키마 검사에서 거절한 모델 출력의 앞부분. 실패 원인 파악용."""
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
    from groq import BadRequestError

    schema = qt.build_schema(fmt)
    for attempt in range(CALL_RETRIES + 1):
        try:
            text = chat(client, messages, QUIZ_TEMPERATURE, QUIZ_MAX_TOKENS, schema)
        except BadRequestError as e:
            # 모델 출력이 스키마 검사에 걸린 경우만 재시도한다. 그 밖의 400은 그대로 올린다
            if "json_validate_failed" not in str(e):
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
    counts: dict[str, int],
    rng: random.Random | None = None,
) -> list[dict]:
    """유형별 개수(counts)만큼 문제를 만든다. 문제마다 문단 하나에서 LLM 호출 한 번이 기본이다.

    거부되거나 검증에 실패하면 같은 출처의 다음 문단(없으면 다른 출처)으로 최대 SLOT_TRIES번 시도한다.
    """
    rng = rng or random.Random()
    pools = {
        ORIGIN_CITED: [candidate_from_source(s) for s in cited],
        ORIGIN_PAGE: same_page_pool(index, question, cited, rng),
    }
    slots = plan_slots(counts, len(pools[ORIGIN_PAGE]), rng)
    used: Counter[str] = Counter()  # 문단을 골고루 쓰려고 쓴 횟수가 적은 것부터 고른다
    quiz: list[dict] = []
    previous: list[str] = []

    for fmt, origin in slots:
        other = ORIGIN_PAGE if origin == ORIGIN_CITED else ORIGIN_CITED
        order = [
            *sorted(pools[origin], key=lambda c: used[c["unit_id"]]),
            *sorted(pools[other], key=lambda c: used[c["unit_id"]]),
        ]
        for cand in order[:SLOT_TRIES]:
            q, reason = try_question(client, fmt, question, cand, previous)
            if q is None:
                logger.info("[%s] %s 문단 실패: %s", fmt, cand["origin"], reason)
                continue
            used[cand["unit_id"]] += 1
            previous.append(q.get("statement") or q["question"])
            quiz.append(q)
            break
        else:
            logger.warning("%s 문제를 만들지 못함 (%d번 시도)", fmt, SLOT_TRIES)

    for q in quiz:
        if q["format"] == "mcq":
            qt.shuffle_mcq(q, rng)
    if len(quiz) < sum(counts.values()):
        logger.warning(
            "요청 %d문제 중 %d문제만 만들었습니다", sum(counts.values()), len(quiz)
        )
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
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG 기반 답변 + 확인 퀴즈 (Groq)")
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
        default=list(qt.FORMATS),
        help=f"퀴즈 유형 (쉼표로 구분, 기본: {','.join(qt.FORMATS)})",
    )
    parser.add_argument(
        "--quiz", type=int, default=4, help="총 문제 수 (유형별로 나눔)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="문단 고르기·보기 섞기 시드 (재현용)"
    )
    parser.add_argument("--no-quiz", action="store_true")
    parser.add_argument("--solve", action="store_true", help="퀴즈를 직접 풀기")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from groq import APIConnectionError, APIStatusError

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
            counts = qt.split_counts(args.formats, args.quiz)
            questions = make_quiz(
                client,
                args.question,
                quiz_sources,
                index,
                counts,
                random.Random(args.seed),
            )
            print_quiz(questions, args.solve)
    except APIConnectionError as e:
        logger.error("Groq 연결 실패: %s", e)
    except APIStatusError as e:
        logger.error("Groq API 오류 (%s): %s", e.status_code, e.message)
    except QuizFormatError as e:
        logger.error("%s", e)


if __name__ == "__main__":
    main()
