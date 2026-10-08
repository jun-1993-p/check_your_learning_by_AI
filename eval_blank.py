"""RAG on/off 빈칸 문제 정확도 테스트 (A+ 문단에 근거해 새 문장으로 문제 생성).

설계: .idea_folder/정확도_2차_근거문장_인덱스_설계.md (0장). 문장 방식과 단위 A·B는 폐기했다.

케이스(책, 개념) 하나가 결과 CSV의 한 줄이고, 한 줄에 on과 off를 모두 담는다. on을 먼저 돌리고,
on이 찾은 근거 URL을 off에 넘긴다.
- on : text_paragraphs.ParagraphIndex.find()의 점수순 A+ 문단 K개(--max-tries, 기본 3)에서 문제를 전부 만든다.
       LLM은 문단에 기반해 새 문장(빈칸 `____` 포함)과 정답을 쓰거나 사유와 함께 거부한다.
       만든 문제는 문제 문장만 보고(문단·개념 없이) 풀 수 있는지 검증한다: 풀이 LLM이 정답을 하나로
       정할 수 있다고 답하고 정답과 같은 뜻이어야 통과다(글자가 다르면 LLM이 동의 여부를 판정).
       통과한 문제가 여럿이면 규칙 순위로 하나만 쓴다: 풀이 답이 정답과 글자까지 같은 것 →
       빈칸이 적은 것 → 검색 순위가 높은 것. 그 문단이 그 개념의 근거다. 이미 쓴 문단은 이후 케이스에서
       뺀다. K개가 전부 거부되면 다음 K개로 넘어가고(--rounds, 기본 2라운드), 그래도 없으면
       "출제 못함: 거부 — 문단별 사유"를 적는다. 문단이 K개보다 적으면 있는 만큼만 한다.
       --context on이면 [핵심 문단](1순위, 근거)에 그 문단이 속한 [같은 페이지](2순위,
       참고 맥락)를 붙인다 (기본 off: 입력이 약 2배가 된다).
       코드 검증: 새 문장에 빈칸이 있고, 정답이 핵심 문단에 있어야 한다. 정답이 문제에
       남아 있으면 코드가 함께 가린다.
- off: LLM에 책 제목, on이 찾은 근거 URL, 개념만 주고(웹을 볼 수 없고 문단 본문도 없다) 그 페이지의
       내용을 기억하는 대로 빈칸 문제를 쓰게 한다. 기억나지 않으면 사유와 함께 거부할 수 있다.
       on이 근거를 못 찾았으면 넘길 URL이 없으니 off는 건너뛴다. --skip-off로 전부 건너뛸 수 있다.
       off의 문제는 on이 찾은 "근거 문단"과 견주어 사람이 판정한다 (책 내용에 맞게 썼는가).
- 호출이 실패하면(JSON 생성 오류 등) 같은 입력으로 CALL_RETRIES번 다시 시도하고, 그래도 실패하면
  on은 그 순위를 "호출 실패"로 기록하고 넘어가고, off는 "(오류: …)"로 적는다.

결과 CSV 열 (14개): id, book_id, off 문제, off 답, on 문제, on 답, 근거 문단, 근거 URL,
그리고 사람이 채우는 판정 3열. 값은 "(on,off)" 형식의 o/x, 예: "(o,x)". 해당 없으면 "-":
    근거 확인: 문제가 근거 문단에 나온 사실에 맞는가 (설계 문서 6장 기준)
    답 유효성: 빈칸에 들어갈 답이 하나로 정해지고 맞는가
    거부 타당: LLM이 거부한 칸만. 거부가 맞았는가
출제하지 못한 칸은 "문제"에 사유를 적는다: "(출제 못함: 거부 — 1:정보 없음 / 2:정보 없음)",
"(거부: 정보 없음)", "(건너뜀: …)", "(오류: …)". 오류가 있는 줄만 --resume에서 다시 실행한다.

--dry-run은 LLM 없이 검색을 확인하고, book_id 필터가 실제로 그 책의 문단만 찾는지 검사한다
(필터 없이 검색한 결과와 비교). 필터 검사 결과는 .data/eval/filter_check_*.csv에 쓴다.

결과는 .data/eval/blank_YYYYMMDD_HHMMSS_모델_ctx….csv에 한 줄씩 바로 쓴다 (중간에 끊겨도 남는다).
LLM 하루 한도에 걸리면 그때까지의 결과만 저장하고, --resume으로 이어 실행한다.

사용 예:
    python eval_blank.py --dry-run                           # LLM 없이 검색·필터 검사
    python eval_blank.py --ids 1 16 18                       # 일부 케이스
    python eval_blank.py --skip-off --ids 1 3 4              # off 없이 on만
    python eval_blank.py --context on --ids 16               # 페이지 맥락을 줘서 (기본은 off)
    python eval_blank.py --resume .data/eval/blank_20261007_1200_gpt-oss-20b_ctxoff.csv
"""

import argparse
import builtins
import csv
import json
import keyword
import logging
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

import quiz_session as qs
from eval_common import (
    CASES_CSV,
    GROQ_MAX_RETRIES,
    MAX_API_ERRORS,
    OUT_DIR,
    TEMPERATURE,
    ApiErrorsPiledUp,
    DailyLimitReached,
    load_cases,
    model_options,
    parse_book_id,
    titles_by_book_id,
)
from quiz_templates import BLANK, BlankError, contains, make_question
from text_embed import embed, load_chunk_index, load_model
from text_paragraphs import ParagraphIndex, get_collection

# 예외 클래스는 고른 provider(llm_groq / llm_nvidia)의 것을 쓴다
APIError = qs.APIError
RateLimitError = qs.RateLimitError

MAX_TRIES = 3  # 한 라운드에서 한꺼번에 문제를 만들어 보는 문단 수 (K)
MAX_ROUNDS = 2  # 한 라운드가 전부 거부되면 다음 K개로 넘어가는 최대 라운드 수
FIX_TRIES = 1  # 코드 검증에 실패하면 이유를 알려 주고 같은 문단으로 고쳐 쓰게 하는 횟수
CALL_RETRIES = 2  # 호출이 실패(JSON 생성 오류 등)하면 같은 입력으로 다시 시도하는 횟수
FREE_SEARCH_K = 5  # 필터 검사: 필터 없이 검색해 볼 상위 문단 수

HANGUL = re.compile(r"[가-힣]")
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
LATIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PYTHON_NAMES = set(keyword.kwlist) | set(dir(builtins))
# 기본 자료형의 메서드 이름 (is_integer, bit_length 등). 밑줄이 있어도 예제용 이름이 아니다
API_NAMES = set().union(
    *(dir(t) for t in (str, bytes, list, tuple, dict, set, frozenset, int, float, complex, bool, object))
)
MIN_HANGUL_CHARS = 5  # 문제 문장에 이 이상 한글이 있어야 한국어 문제로 본다

OK_REASON = "없음"
REJECT_REASONS = [
    "정보 없음",
    "비유",
    "다른 언어",
    "추상적 주장",
    "개념 불일치",
]
SOURCE_PARAGRAPH, SOURCE_PAGE, SOURCE_NONE = "문단", "페이지", "없음"
# 칸의 상태
MADE, NOT_MADE, FAILED, SKIPPED = "출제", "출제 못함", "오류", "건너뜀"
CELL_PREFIX = {
    NOT_MADE: ("(출제 못함", "(거부"),
    FAILED: ("(오류",),
    SKIPPED: ("(건너뜀",),
}

PARAGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "reject_reason": {"type": "string", "enum": [OK_REASON, *REJECT_REASONS]},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "source": {
            "type": "string",
            "enum": [SOURCE_PARAGRAPH, SOURCE_PAGE, SOURCE_NONE],
        },
    },
    "required": ["reject_reason", "question", "answer", "source"],
    "additionalProperties": False,
}
OFF_SCHEMA = {
    "type": "object",
    "properties": {
        "reject_reason": {"type": "string", "enum": [OK_REASON, *REJECT_REASONS]},
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["reject_reason", "question", "answer"],
    "additionalProperties": False,
}
PARAGRAPH_SYSTEM = f"""너는 파이썬 학습 퀴즈 출제자야.
아래 [개념]을 확인하는 빈칸 채우기 문제의 문장을 [핵심 문단]에 근거해서 새로 써 줘.
- [핵심 문단]이 1순위 근거야. 문제 내용과 정답은 반드시 [핵심 문단]에 있는 내용이어야 하고, 문단 밖 지식은 쓰지 마.
- [같은 페이지]는 2순위 참고 자료야. 그 안의 {"<<핵심 문단>>"} 자리가 [핵심 문단]이야. 정의·예제 이름·앞뒤 코드를 이해하는 데만 쓰고, 정답이나 문제 내용을 여기에서만 가져오지 마. [같은 페이지]가 없으면 무시해.
- question: 학습자가 [핵심 문단]을 보지 않고도 이해할 수 있는 자기완결적인 한 문장. 반드시 한국어로 써 (영어 문장 금지, 코드 이름만 영어 가능).문단의 문장을 그대로 베끼지 말고 직접 써. 문단 안에서 정의되지 않은 변수·값 이름은 쓰지 마. 접속어로 시작하지 마.
  정답이 들어갈 자리는 "{BLANK}"로 쓰고, 같은 단어가 여러 번 나오면 모두 "{BLANK}"로 가려.
- answer: {BLANK}에 들어갈 파이썬 개념 용어(조사 제외). 문단에 '변수(variable)'처럼 한글 용어가 있으면 한글로 쓰고, 영어는 while·and·len처럼 코드에 쓰는 이름일 때만 써. 코드의 실행 결과나 값을 묻는 문제는 만들지 마. '수정할 수 없다'처럼 서술어 전체를 정답으로 하지 말고 핵심 용어(수정)만 빈칸으로 만들어서 나머지는 문제에 남겨. 반드시 띄어쓰기 없는 한 단어여야 하고, 구나 문장은 안 돼. [핵심 문단]에도 나와야 하고, question의 다른 곳에는 남아 있으면 안 돼.
- source: 문제 내용의 근거가 [핵심 문단]이면 "{SOURCE_PARAGRAPH}", [같은 페이지]에서만 나오면 "{SOURCE_PAGE}".
거부 사유 (문제를 낼 수 없을 때 하나 골라):
- 정보 없음: 문단이 도입·예고·안내·잡담·감정·책 이야기뿐
- 비유: 문단 전체가 일상 사물에 빗댄 설명이거나, 문제로 만들 수 있는 내용이 "~ 같은 존재", "~처럼"처럼 비유 표현뿐이라 빈칸의 답이 비유 표현이 될 수밖에 없음
- 다른 언어: 파이썬이 아닌 언어 이야기
- 추상적 주장: 확인할 수 있는 사실이 없음 (느낌·평가·감상, 예: "~는 장벽이다", "~는 중요하다")
- 개념 불일치: 문단이 [개념]을 다루지 않음
출제하면 reject_reason은 "{OK_REASON}", 거부하면 question·answer는 빈 문자열, source는 "{SOURCE_NONE}".
JSON 객체 하나만 출력해."""
OFF_SYSTEM = f"""너는 파이썬 학습 퀴즈 출제자야.
학습자가 아래 [책]의 [페이지 URL]에 있는 페이지에서 아래 [개념]을 공부했어. 그 페이지의 내용에 기반한 빈칸 채우기 문제의 문장 하나를 새로 써 줘.
- 너는 웹을 볼 수 없고 페이지 본문도 받지 못했어. 책 내용을 기억하는 대로 써.
- question: 개념을 확인하는 한 문장. 반드시 한국어로 써 (영어 문장 금지, 코드 이름만 영어 가능). 정답이 들어갈 자리는 "{BLANK}"로 쓰고, 같은 단어가 여러 번 나오면 모두 "{BLANK}"로 가려.
- answer: {BLANK}에 들어갈 파이썬 개념 용어(조사 제외). 반드시 띄어쓰기 없는 한 단어여야 하고, 구나 문장은 안 돼.
- 그 페이지의 내용이 기억나지 않아 문제를 낼 수 없으면 거부해. 거부 사유(하나 골라): 정보 없음 / 비유 / 다른 언어 / 추상적 주장 / 개념 불일치.
출제하면 reject_reason은 "{OK_REASON}", 거부하면 question·answer는 빈 문자열.
JSON 객체 하나만 출력해."""

SOLVE_SYSTEM = """너는 파이썬을 공부하는 학습자야. 아래 [문제]의 빈칸 ____에 들어갈 말을 풀어. 문제 문장 말고는 아무 자료도 없어.
- answer: ____에 들어갈 파이썬 개념 용어 하나(띄어쓰기 없는 한 단어, 조사 제외). 빈칸이 여러 개면 같은 말이 모두 들어가.
- determinable: 문제 문장만 읽고 정답을 하나로 정할 수 있으면 true. 문제에 정의되지 않은 이름·값이 나오거나 들어갈 수 있는 말이 여럿이라 정할 수 없으면 false이고, answer에는 가장 가능성 높은 말을 써.
JSON 객체 하나만 출력해."""
SOLVE_SCHEMA = {
    "type": "object",
    "properties": {"determinable": {"type": "boolean"}, "answer": {"type": "string"}},
    "required": ["determinable", "answer"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = """너는 [문제]의 빈칸에 [답 1]과 [답 2] 중 어느 쪽을 넣어도 같은 개념을 말하는지 판단해.
- 동의어, 한영 표기, 접미 차이(while / while문, 변수 / variable)는 같은 것으로 봐.
- 더 넓거나 좁은 개념이거나 다른 대상이면 다른 것으로 봐.
JSON 객체 {"same": true 또는 false} 하나만 출력해."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
    "additionalProperties": False,
}

FIELDS = [
    "id",
    "book_id",
    "off 문제",
    "off 답",
    "on 문제",
    "on 답",
    "근거 문단",
    "근거 URL",
    "근거 확인",
    "답 유효성",
    "거부 타당",
]
FILTER_FIELDS = [
    "id",
    "book_id",
    "개념",
    "필터 문단 책",
    "필터 통과",
    "무필터 상위 문단 책",
    "무필터 1위 다른 책",
    "무필터 상위 다른 책 수",
    "무필터 상위 문단 수",
]

logger = logging.getLogger("eval_blank")


def cell_status(text: str) -> str:
    """ "문제" 칸의 상태. 문제 문장이면 MADE, 사유 문구면 그 사유의 종류, 비어 있으면 SKIPPED."""
    if not text:
        return SKIPPED
    for status, prefixes in CELL_PREFIX.items():
        if text.startswith(prefixes):
            return status
    return MADE


# ---------------------------------------------------------------- LLM


def log_rate_limit(response: httpx.Response) -> None:
    """429 응답의 한도 정보를 남긴다. SDK가 조용히 재시도해서, 어떤 한도(분당 토큰·요청)에 걸렸는지 안 보인다."""
    if response.status_code != 429:
        return
    response.read()
    h = response.headers
    try:
        message = response.json().get("error", {}).get("message", "")
    except ValueError:
        message = response.text
    logger.info(
        "429 한도: %s | retry-after=%s, 남은 토큰=%s/%s, 남은 요청=%s/%s",
        message[:200], h.get("retry-after"),
        h.get("x-ratelimit-remaining-tokens"), h.get("x-ratelimit-limit-tokens"),
        h.get("x-ratelimit-remaining-requests"), h.get("x-ratelimit-limit-requests"),
    )  # fmt: skip


def call_json(client, model: str, system: str, user: str, schema: dict) -> dict:
    options = model_options(model)
    max_tokens = options.pop("max_tokens")
    try:
        text = qs.chat(
            client,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            TEMPERATURE,
            max_tokens,
            schema,
            model=model,
            **options,
        )
    except RateLimitError as e:
        if qs.llm.is_daily_limit(e):
            raise DailyLimitReached(str(e)) from e
        raise
    match = qs.JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"JSON 없음: {text[:120]}")
    return json.loads(match.group())  # JSONDecodeError는 ValueError의 하위 클래스


def parse_output(data: dict) -> dict:
    """LLM 응답을 읽는다. 거부 사유가 목록에 없으면 ValueError."""
    out = {
        "reject_reason": str(data.get("reject_reason", "")),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
    }
    if out["reject_reason"] != OK_REASON and out["reject_reason"] not in REJECT_REASONS:
        raise ValueError(f"알 수 없는 거부 사유: {out['reject_reason']!r}")
    return out


api_error_streak = 0  # 연속으로 난 API 오류 수. 호출이 한 번 성공하면 0으로 돌아간다


def with_retries(call, tag: str, usage: dict, chars: int):
    """call()을 최대 CALL_RETRIES+1번 시도한다. 모두 실패하면 None. 호출 수·입력 글자 수를 센다.

    API 오류가 MAX_API_ERRORS번 연속으로 쌓이면 ApiErrorsPiledUp으로 멈춘다.
    """
    global api_error_streak
    for attempt in range(1, CALL_RETRIES + 2):
        usage["calls"] += 1
        usage["chars"] += chars
        try:
            result = call()
            api_error_streak = 0
            return result
        except APIError as e:
            api_error_streak += 1
            logger.warning(
                "[%s] API 오류 %d회 연속 (%d/%d): %s",
                tag, api_error_streak, attempt, CALL_RETRIES + 1, e,
            )  # fmt: skip
            if api_error_streak >= MAX_API_ERRORS:
                raise ApiErrorsPiledUp(str(e)) from e
        except ValueError as e:
            api_error_streak = 0  # 서버는 응답했다 (JSON 형식 문제 등)
            logger.warning(
                "[%s] 호출 실패 (%d/%d): %s", tag, attempt, CALL_RETRIES + 1, e
            )
    return None


def paragraph_prompt(concept: str, unit: dict, context: str | None) -> str:
    user = f"[개념]\n{concept}\n\n[핵심 문단]\n{unit['text']}"
    if context is not None:
        user += f"\n\n[같은 페이지]\n{context}"
    return user


def check_one_word(answer: str) -> None:
    """정답은 띄어쓰기 없는 한 단어여야 한다 (여러 단어면 너무 어려워진다)."""
    if len(answer.split()) != 1:
        raise BlankError(f"정답이 한 단어가 아님: {answer}")
    if answer.endswith(("다", "요")):
        raise BlankError(
            f"정답이 서술어로 끝남: {answer} (핵심 용어만 빈칸으로 만들고 '~할 수 없다' 같은 서술어는 문제에 남길 것)"
        )


def check_korean(question: str) -> None:
    """문제 문장은 한국어여야 한다 (정답·코드 이름만 영어). 한글이 거의 없으면 영어 문제로 본다."""
    if len(HANGUL.findall(question)) < MIN_HANGUL_CHARS:
        raise BlankError(f"한국어 문제가 아님: {question[:40]}")


def check_korean_term(answer: str, unit_text: str) -> None:
    """영어 정답은 문단의 설명(코드 블록 밖)에 쓰인 말이어야 한다. '변수(variable)'처럼 한글 용어가 있으면 한글로 내야 한다."""
    if not re.fullmatch(r"[A-Za-z]+", answer):
        return
    if re.search(rf"[가-힣]\s*\(\s*{re.escape(answer)}\s*\)", unit_text, re.IGNORECASE):
        raise BlankError(f"한글 용어가 있는데 영어 정답: {answer}")
    prose = FENCED_CODE.sub("", unit_text)
    if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(answer)}(?![A-Za-z0-9_])", prose, re.IGNORECASE):
        raise BlankError(f"영어 정답이 코드에만 있음: {answer}")


def check_defined_names(question: str, answer: str, unit_text: str) -> None:
    """문제나 정답이 문단의 코드에서 만든 이름(변수·함수·클래스)이면 안 된다. 학습자는 그 코드를 못 본다."""
    for name in set(LATIN_NAME.findall(f"{question} {answer}")) - PYTHON_NAMES:
        escaped = re.escape(name)
        if re.search(
            rf"\b{escaped}\s*=(?!=)|\b(def|class)\s+{escaped}\b|\bfor\s+{escaped}\s+in\b",
            unit_text,
        ):
            raise BlankError(f"문단 코드에서 만든 이름이 나옴: {name}")
    # 문단 설명에서만 정의된 예제용 이름(num_accounts 같은 snake_case)은 코드 블록 밖에 있어도 안 된다
    for name in set(LATIN_NAME.findall(question)) - PYTHON_NAMES - API_NAMES:
        if "_" in name.strip("_") and not name.startswith("__"):
            raise BlankError(f"예제용 이름이 문제에 나옴: {name} (학습자는 그 예제를 못 봄)")


def validate_paragraph(out: dict, unit_text: str) -> str:
    """출제 결과를 코드로 검증해 빈칸 문제를 돌려준다. 실패하면 BlankError.

    정답이 핵심 문단에 있는지를 코드가 직접 확인한다.
    """
    check_one_word(out["answer"])
    check_korean(out["question"])
    check_korean_term(out["answer"], unit_text)
    check_defined_names(out["question"], out["answer"], unit_text)
    question, _ = make_question(out["question"], out["answer"])
    if not contains(unit_text, out["answer"]):
        raise BlankError("정답이 핵심 문단에 없음")
    return question


# ---------------------------------------------------------------- 필터 검사


def check_filter(pi: ParagraphIndex, cases: list[dict]) -> list[dict]:
    """book_id를 주면 그 책의 문단만 찾는가. 필터 없는 검색과 비교해 검사가 의미 있는지도 본다."""
    results = []
    for case in cases:
        concept, book_id = case["개념"], parse_book_id(case["book_id"])
        found = pi.find(concept, book_id)
        books = {u["book_id"] for u in found}
        free = pi.collection.query(
            query_embeddings=embed(pi.model, [concept]),
            n_results=FREE_SEARCH_K,
            include=["metadatas"],
        )
        free_books = [m["book_id"] for m in free["metadatas"][0]]
        results.append(
            {
                "id": case["id"].strip(),
                "book_id": book_id,
                "개념": concept,
                "필터 문단 책": sorted(books, key=str),
                "필터 통과": "o" if books <= {book_id} else "x",
                "무필터 상위 문단 책": free_books,
                "무필터 1위 다른 책": (
                    "o" if free_books and free_books[0] != book_id else "x"
                ),
                "무필터 상위 다른 책 수": sum(b != book_id for b in free_books),
                "무필터 상위 문단 수": len(free_books),
            }
        )
    return results


def summarize_filter(rows: list[dict]) -> None:
    passed = sum(r["필터 통과"] == "o" for r in rows)
    first_other = sum(r["무필터 1위 다른 책"] == "o" for r in rows)
    other = sum(r["무필터 상위 다른 책 수"] for r in rows)
    total = sum(r["무필터 상위 문단 수"] for r in rows)
    print("\n=== book_id 필터 검사 ===")
    print(f"  필터 통과 {passed}/{len(rows)}  " + str(
        [r["id"] for r in rows if r["필터 통과"] == "x"]
    ))  # fmt: skip
    print(
        f"  (대조) 필터 없이 검색: 1위가 다른 책 {first_other}/{len(rows)}, "
        f"상위 문단 중 다른 책 {other}/{total}"
    )


def write_filter_check(rows: list[dict], stamp: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"filter_check_{stamp}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FILTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------- 케이스 실행


def fix_prompt(base: str, out: dict, error: BlankError) -> str:
    """검증에 실패한 문제와 이유를 원래 프롬프트 뒤에 붙여 고쳐 쓰게 한다."""
    return (
        f"{base}\n\n[이전에 쓴 문제]\n문제: {out['question']}\n정답: {out['answer']}"
        f"\n\n[고칠 점]\n{error}\n위 문제를 이 문제점만 고쳐서 다시 써 줘. 고칠 수 없으면 거부해."
    )


def generate_on(
    client, model, concept: str, unit: dict, context: str | None, tag: str, usage: dict
) -> tuple[dict | None, str, str]:
    """한 문단으로 문제를 만든다. 코드 검증에 실패하면 이유를 알려 주고 FIX_TRIES번까지 고쳐 쓰게 한다.

    돌려주는 값: (LLM 응답, 거부·실패 사유 또는 OK_REASON, 완성된 문제).
    LLM이 스스로 거부한 경우는 고쳐 쓸 문제가 없으니 다시 시키지 않는다.
    """
    base = paragraph_prompt(concept, unit, context)
    user, out, reason = base, None, "호출 실패"
    for attempt in range(FIX_TRIES + 1):
        out = with_retries(
            lambda user=user: parse_output(
                call_json(client, model, PARAGRAPH_SYSTEM, user, PARAGRAPH_SCHEMA)
            ),
            tag if attempt == 0 else f"{tag} 고쳐 쓰기 {attempt}",
            usage,
            len(PARAGRAPH_SYSTEM) + len(user),
        )
        if out is None:
            return None, "호출 실패", ""
        if out["reject_reason"] != OK_REASON:
            return out, out["reject_reason"], ""
        try:
            return out, OK_REASON, validate_paragraph(out, unit["text"])
        except BlankError as e:
            reason = f"검증 실패({e})"
            user = fix_prompt(base, out, e)
    return out, reason, ""


def same_text(a: str, b: str) -> bool:
    return re.sub(r"\s+", "", a).lower() == re.sub(r"\s+", "", b).lower()


def verify_question(
    client, model, question: str, gold: str, tag: str, usage: dict
) -> tuple[bool, bool, str]:
    """문제 문장만 보고 풀 수 있는지 검증한다. (통과, 정답과 글자까지 같음, 실패 사유).

    풀이 LLM에는 문단·개념 없이 문제 문장만 준다. 정할 수 없다고 하거나 답이 다르면 실패다.
    글자가 다르면 LLM이 같은 뜻인지(동의어·표기 차이) 판정한다.
    """
    solved = with_retries(
        lambda: call_json(
            client, model, SOLVE_SYSTEM, f"[문제]\n{question}", SOLVE_SCHEMA
        ),
        f"{tag} 풀이",
        usage,
        len(SOLVE_SYSTEM) + len(question),
    )
    if solved is None:
        return False, False, "풀이 호출 실패"
    got = str(solved.get("answer", "")).strip()
    if solved.get("determinable") is not True or not got:
        return False, False, "문제만으로 정답을 정할 수 없음"
    if same_text(got, gold):
        return True, True, ""
    user = f"[문제]\n{question}\n\n[답 1]\n{gold}\n\n[답 2]\n{got}"
    judged = with_retries(
        lambda: call_json(client, model, JUDGE_SYSTEM, user, JUDGE_SCHEMA),
        f"{tag} 동의 판정",
        usage,
        len(JUDGE_SYSTEM) + len(user),
    )
    if judged is not None and judged.get("same") is True:
        return True, False, ""
    return False, False, f"풀이 답 불일치: {got}"


def run_on(
    client,
    model,
    pi: ParagraphIndex,
    case: dict,
    used: set[str],
    tries: int,
    with_context: bool,
    rounds: int = MAX_ROUNDS,
    solver_model: str | None = None,
) -> dict:
    """on: 점수순 문단 tries개로 문제를 만들고 문제만 읽어 풀 수 있는 것 중 하나를 고른다.

    전부 거부되면 다음 tries개로 넘어가고(최대 rounds라운드), 그래도 없으면 거부 사유를 적는다.
    통과한 문제가 여럿이면 규칙 순위로 하나만 쓴다: 풀이 답이 정답과 글자까지 같은 것 →
    빈칸이 적은 것 → 검색 순위가 높은 것. 결과 dict: status, 문제, 답, 문단, URL, unit_id, 순위, 거부 목록.
    """
    book_id, concept = case["book_id"], case["개념"]
    solver = solver_model or model
    usage = {"calls": 0, "chars": 0}
    result = {"usage": usage, "rejects": [], "rank": None}
    found = pi.find(concept, book_id, frozenset(used), fit=True)
    if not found:
        return {**result, "status": FAILED, "문제": "(오류: 근거 문단 없음)"}
    for round_no in range(rounds):
        batch = found[round_no * tries : (round_no + 1) * tries]
        if not batch:
            break
        passed = []
        for rank, unit in enumerate(batch, start=round_no * tries + 1):
            tag = f"{case['id']}/on {rank}위"
            context = pi.page_context(unit) if with_context else None
            out, reason, question = generate_on(
                client, model, concept, unit, context, tag, usage
            )
            if reason == OK_REASON:
                ok, exact, why = verify_question(
                    client, solver, question, out["answer"], tag, usage
                )
                if not ok:
                    reason = f"풀이 검증 실패({why})"
            if reason != OK_REASON:
                result["rejects"].append(f"{rank}:{reason}")
                logger.info(
                    "[%s/on] %d위 거부 %s: %s",
                    case["id"], rank, reason, " ".join(unit["text"].split())[:60],
                )  # fmt: skip
                continue
            passed.append((not exact, question.count(BLANK), rank, unit, out, question))
        if passed:
            _, _, rank, unit, out, question = min(passed, key=lambda p: p[:3])
            used.add(unit["unit_id"])
            logger.info(
                "[%s/on] 통과 %d개 중 %d위 선택", case["id"], len(passed), rank
            )
            return {
                **result,
                "status": MADE,
                "문제": question,
                "답": out["answer"],
                "문단": unit["text"],
                "URL": unit["url"],
                "unit_id": unit["unit_id"],
                "rank": rank,
            }
    reasons = " / ".join(result["rejects"])
    return {**result, "status": NOT_MADE, "문제": f"(출제 못함: 거부 — {reasons})"}


def run_off(client, model, book: str, url: str, concept: str) -> dict:
    """off: 책 제목·on의 근거 URL·개념만 주고 기억으로 문제를 쓰게 한다 (문단 본문은 주지 않는다)."""
    usage = {"calls": 0, "chars": 0}
    user = f"[책]\n{book}\n\n[페이지 URL]\n{url}\n\n[개념]\n{concept}"
    out = with_retries(
        lambda: parse_output(call_json(client, model, OFF_SYSTEM, user, OFF_SCHEMA)),
        f"{concept}/off",
        usage,
        len(OFF_SYSTEM) + len(user),
    )
    result = {"usage": usage}
    if out is None:
        return {**result, "status": FAILED, "문제": "(오류: 호출 실패)"}
    if out["reject_reason"] != OK_REASON:
        return {
            **result,
            "status": NOT_MADE,
            "문제": f"(거부: {out['reject_reason']})",
        }
    try:
        check_one_word(out["answer"])
        check_korean(out["question"])
        question, _ = make_question(out["question"], out["answer"])
    except BlankError as e:
        return {**result, "status": NOT_MADE, "문제": f"(출제 못함: 검증 실패({e}))"}
    return {**result, "status": MADE, "문제": question, "답": out["answer"]}


def build_row(case: dict, on: dict, off: dict | None, skip_off: bool) -> dict:
    """한 케이스의 on·off 결과를 CSV 한 줄로. off=None이면 on이 근거를 못 찾아 건너뛴 것이다."""
    if skip_off:
        off_cells = {"off 문제": "", "off 답": ""}
    elif off is None:
        off_cells = {"off 문제": "(건너뜀: on에서 근거를 못 찾음)", "off 답": ""}
    else:
        off_cells = {"off 문제": off["문제"], "off 답": off.get("답", "")}
    return {
        "id": case["id"],
        "book_id": case["book_id"],
        **off_cells,
        "on 문제": on["문제"],
        "on 답": on.get("답", ""),
        "근거 문단": on.get("문단", ""),
        "근거 URL": on.get("URL", ""),
        # 아래는 CSV에 쓰지 않는 내부 값 (요약용). FIELDS 밖의 키는 writer가 버린다
        "_on": on,
        "_off": off,
    }


# ---------------------------------------------------------------- 이어 실행


def has_error(row: dict) -> bool:
    return FAILED in (cell_status(row["on 문제"]), cell_status(row["off 문제"]))


def load_finished(path: Path) -> list[dict]:
    """이전 결과에서 오류 없는 줄만. 오류가 있는 줄은 다시 실행하도록 버린다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"--resume 파일의 열이 현재 형식과 다릅니다: {sorted(missing)}"
            )
        return [r for r in reader if not has_error(r)]


# ---------------------------------------------------------------- 요약


def summarize(rows: list[dict], resumed: int) -> None:
    print(f"\n=== 결과 (n={len(rows)}) ===")
    for label, key in (("on", "on 문제"), ("off", "off 문제")):
        counts = Counter(cell_status(r[key]) for r in rows)
        print(f"  {label:<3}: " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    fresh = [
        r for r in rows if "_on" in r
    ]  # 이어 실행으로 가져온 줄은 순위·호출 수를 모른다
    ons = [r["_on"] for r in fresh]
    ranks = Counter(o["rank"] for o in ons if o["rank"])
    if ranks:
        print(
            "  on 출제 순위: " + "  ".join(f"{k}위 {ranks[k]}" for k in sorted(ranks))
        )
    attempts = Counter()
    for o in ons:
        for item in o["rejects"]:
            attempts[item.split(":", 1)[1].split("(")[0]] += 1
    n_try = sum(attempts.values()) + sum(o["status"] == MADE for o in ons)
    if attempts:
        print(f"  on 거부 사유 (전체 시도 {n_try}번 중):")
        for reason, n in attempts.most_common():
            print(f"    {reason}: {n} ({n / n_try:.0%})")
    usages = [o["usage"] for o in ons] + [
        r["_off"]["usage"] for r in fresh if r["_off"]
    ]
    if usages:
        calls = sum(u["calls"] for u in usages)
        chars = sum(u["chars"] for u in usages)
        per_case = statistics.mean(
            (
                r["_on"]["usage"]["calls"]
                + (r["_off"]["usage"]["calls"] if r["_off"] else 0)
            )
            for r in fresh
        )
        print(
            f"  호출 수 합 {calls} (케이스당 평균 {per_case:.1f}), 입력 글자 수 합 {chars:,}"
        )
    plen = [len(o["문단"]) for o in ons if o["status"] == MADE]
    if plen:
        print(
            f"  on 근거 문단 글자 수: 평균 {statistics.mean(plen):.0f} ({min(plen)}~{max(plen)})"
        )
    if resumed:
        print(
            f"  (이어 실행으로 가져온 {resumed}줄은 순위·호출 수·거부 사유 집계에서 빠짐)"
        )


# ---------------------------------------------------------------- 실행


def main() -> None:
    load_dotenv(override=True)  # 셸에 남은 환경변수보다 .env가 우선
    parser = argparse.ArgumentParser(description="RAG on/off 빈칸 문제 정확도 테스트")
    parser.add_argument("--cases", type=Path, default=CASES_CSV)
    parser.add_argument("--ids", nargs="+", help="실행할 케이스 id (생략하면 전체)")
    parser.add_argument(
        "--context",
        choices=["on", "off"],
        default="off",
        help="[같은 페이지] 맥락을 줄지 (기본: off). 비교하려면 두 번 따로 돌린다",
    )
    parser.add_argument(
        "--skip-off",
        action="store_true",
        help="RAG off(대조군)는 건너뛰고 on만 실행",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=MAX_TRIES,
        help="on에서 한 라운드에 한꺼번에 문제를 만들어 볼 문단 수 (K)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=MAX_ROUNDS,
        help="한 라운드가 전부 거부되면 다음 K개 문단으로 넘어가는 최대 라운드 수",
    )
    parser.add_argument(
        "--solver-model",
        default=None,
        help="문제만 읽고 푸는 검증 LLM. 생략하면 문제 생성 모델과 같다",
    )
    parser.add_argument(
        "--model", default=None, help="문제 생성 LLM 모델 ID. 생략하면 .env의 모델(GROQ_MODEL 또는 NVIDIA_MODEL)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="이전 결과 CSV. 오류 없이 끝난 케이스는 건너뛴다. 같은 --context로 이어야 한다",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="LLM 없이 검색과 book_id 필터 검사만"
    )
    args = parser.parse_args()
    llm_model = args.model or qs.get_model()
    print(f"모델: {llm_model} ({'--model' if args.model else '.env'}, {qs.llm.NAME})")
    with_context = args.context == "on"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    all_cases = load_cases(args.cases)
    cases = all_cases
    if args.ids:
        cases = [c for c in all_cases if c["id"].strip() in set(args.ids)]
    chunk_index = load_chunk_index()
    titles = titles_by_book_id(chunk_index)
    pi = ParagraphIndex(load_model(), get_collection(), chunk_index)
    stamp = f"{datetime.now().astimezone():%Y%m%d_%H%M%S}"

    if args.dry_run:
        for case in cases:
            cid, book_id = case["id"].strip(), parse_book_id(case["book_id"])
            found = pi.find(case["개념"], book_id)
            logger.info(
                "[%s] book %s %s → 문단 %d개, 1위: %s",
                cid, book_id, case["개념"], len(found),
                " ".join(found[0]["text"].split())[:80] if found else "",
            )  # fmt: skip
        filter_rows = check_filter(pi, cases)
        summarize_filter(filter_rows)
        print(f"\n결과: {write_filter_check(filter_rows, stamp)}")
        return

    client = qs.get_client(max_retries=GROQ_MAX_RETRIES, hooks=[log_rate_limit])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.resume:
        # 새 파일을 열기 전에 먼저 읽는다. 이전 파일은 그대로 두고 끝난 줄만 옮긴다
        rows = load_finished(args.resume)
        logger.info("이어서 실행: 끝난 %d줄을 가져옴 (%s)", len(rows), args.resume)
    resumed = len(rows)
    done = {r["id"] for r in rows}
    # 책별로 이미 쓴 문단. 이어 실행이면 결과 파일의 "근거 문단" 글로 문단을 찾아 복원한다
    used: dict[int, set[str]] = {}
    for r in rows:
        if r["근거 문단"]:
            book_id = int(r["book_id"])
            by_text = {u["text"]: uid for uid, u in pi.rows(book_id).items()}
            if r["근거 문단"] in by_text:
                used.setdefault(book_id, set()).add(by_text[r["근거 문단"]])

    out = OUT_DIR / f"blank_{stamp}_{llm_model.split('/')[-1]}_ctx{args.context}.csv"
    if args.resume and out.resolve() == args.resume.resolve():
        raise SystemExit(f"결과 파일이 --resume 파일과 같습니다: {out}")
    stop_reason: DailyLimitReached | ApiErrorsPiledUp | None = None
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()

        for case in cases:
            cid, concept = case["id"].strip(), case["개념"]
            book_id = parse_book_id(case["book_id"])
            book = titles.get(book_id)
            if book is None:
                logger.error("[%s] 책을 찾지 못함: book_id=%s", cid, case["book_id"])
                continue
            if cid in done:
                continue
            info = {"id": cid, "book_id": book_id, "개념": concept}
            try:
                on = run_on(
                    client,
                    llm_model,
                    pi,
                    info,
                    used.setdefault(book_id, set()),
                    args.max_tries,
                    with_context,
                    args.rounds,
                    args.solver_model,
                )
                off = None
                if on["status"] == MADE and not args.skip_off:
                    off = run_off(client, llm_model, book, on["URL"], concept)
            except (DailyLimitReached, ApiErrorsPiledUp) as e:
                stop_reason = e
                break  # 이 케이스는 저장하지 않는다 → --resume 때 다시 만든다
            except (ValueError, APIError) as e:
                message = f"{type(e).__name__}: {e}"
                logger.warning("[%s] 실패: %s", cid, message)
                on = {
                    "status": FAILED,
                    "문제": f"(오류: {message})",
                    "usage": {"calls": 0, "chars": 0},
                    "rejects": [],
                    "rank": None,
                }
                off = None
            row = build_row(info, on, off, args.skip_off)
            logger.info(
                "[%s] on=%s off=%s | %s",
                cid,
                on["status"],
                off["status"] if off else "-",
                on["문제"],
            )
            rows.append(row)
            writer.writerow(row)
            f.flush()

    if stop_reason:
        if isinstance(stop_reason, ApiErrorsPiledUp):
            logger.error("API 오류가 %d번 연속 나서 중단했습니다: %s", MAX_API_ERRORS, stop_reason)
            why = ".env의 모델·API 키와 서비스 상태를 확인한 뒤 이어서 실행하세요"
        else:
            logger.error("하루 한도에 걸려 중단했습니다: %s", stop_reason)
            why = "하루 한도로 중단. 한도가 풀리면 이어서 실행하세요"
        print(
            f"\n{why}:\n"
            f"  python eval_blank.py --resume {out} --model {llm_model} "
            f"--context {args.context}"
            + (" --skip-off" if args.skip_off else "")
            + (f" --ids {' '.join(args.ids)}" if args.ids else "")
        )
    summarize(rows, resumed)
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
