"""RAG on/off 빈칸 문제 정확도 테스트 (A+ 문단에 근거해 새 문장으로 문제 생성).

설계: .idea_folder/정확도_2차_근거문장_인덱스_설계.md (0장). 문장 방식과 단위 A·B는 폐기했다.

케이스(책, 개념)마다 on·off 한 번씩 문제를 만든다.
- on : text_paragraphs.ParagraphIndex.find()의 점수순 A+ 문단을 1위부터 LLM에 한 문단씩 준다.
       LLM은 문단에 기반해 새 문장(빈칸 `____` 포함)과 정답을 쓰거나 사유와 함께 거부한다.
       거부되면 다음 순위로 (최대 --max-tries번), 처음 출제된 문단 하나를 그 개념의 근거로
       쓴다. 이미 쓴 문단은 이후 케이스에서 뺀다. 5번 모두 거부되면 "전부 거부"로 끝낸다.
       --context on이면 [핵심 문단](1순위, 근거)에 그 문단이 속한 [같은 페이지](2순위,
       참고 맥락)를 붙인다 (기본 off: 입력이 약 2배가 된다).
       코드 검증: 새 문장에 빈칸이 있고, 정답이 핵심 문단에 있어야 한다. 정답이 문제에
       남아 있으면 코드가 함께 가린다. LLM이 말한 근거 위치(문단/페이지)는 기록만 한다.
- off: LLM에 책 URL과 개념만 주고 새 문장·정답과 근거 URL·문단을 쓰게 한다. 코드가 URL의
       페이지가 그 책에 있는지, 제시한 문단이 그 페이지 문단과 비슷한지 확인한다 (대조군).
       --skip-off로 건너뛸 수 있다.
- 호출이 실패하면(JSON 생성 오류 등) 같은 문단으로 --CALL_RETRIES번 다시 시도하고, 그래도
  실패하면 그 순위만 "호출 실패"로 기록하고 넘어간다.

판정 열 (사람이 채운다):
    근거 확인(o/x): 근거 문단이 문제를 뒷받침하는 사실인가 (설계 문서 6장 기준)
    답 유일(o/x): 빈칸에 들어갈 답이 하나로 정해지는가
    핵심 개념(o/x): 빈칸이 그 문단의 핵심 파이썬 개념 용어인가 (지엽적인 단어가 아닌가)
    거부 타당(o/x): on에서 LLM이 거부한 행만. 거부가 맞았는가

--dry-run은 LLM 없이 검색을 확인하고, book_id 필터가 실제로 그 책의 문단만 찾는지 검사한다
(필터 없이 검색한 결과와 비교). 필터 검사 결과는 .data/eval/filter_check_*.csv에 쓴다.

문단 ID는 구글 시트가 날짜로 바꾸지 않도록 "c4307-01#g003"처럼 c를 붙여 쓴다.
결과는 .data/eval/blank_YYYYMMDD_HHMMSS_모델_ctx….csv에 한 줄씩 바로 쓴다 (중간에 끊겨도 남는다).
Groq 하루 한도에 걸리면 그때까지의 결과만 저장하고, --resume으로 이어 실행한다.

사용 예:
    python eval_blank.py --dry-run                           # LLM 없이 검색·필터 검사
    python eval_blank.py --ids 1 16 18                       # 일부 케이스
    python eval_blank.py --skip-off --ids 1 3 4              # off(대조군) 없이
    python eval_blank.py --context on off --ids 16           # 페이지 맥락 유무 비교
    python eval_blank.py --resume .data/eval/blank_20261007_1200_gpt-oss-20b_ctxoff.csv
"""

import argparse
import csv
import json
import logging
import os
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError

import quiz_session as qs
from eval_common import (
    CASES_CSV,
    DEFAULT_EVAL_MODEL,
    GROQ_MAX_RETRIES,
    OUT_DIR,
    TEMPERATURE,
    DailyLimitReached,
    load_cases,
    model_options,
    parse_book_id,
    titles_by_book_id,
)
from text_embed import embed, load_chunk_index, load_model
from text_paragraphs import ParagraphIndex, get_collection

BLANK = "____"
MAX_TRIES = 5
CALL_RETRIES = 2  # 같은 문단으로 호출이 실패(JSON 생성 오류 등)하면 다시 시도하는 횟수
MAX_ANSWER_CHARS = 30
FREE_SEARCH_K = 5  # 필터 검사: 필터 없이 검색해 볼 상위 문단 수
BOOK_URL = "https://wikidocs.net/book/{book_id}"
ID_PREFIX = "c"  # 구글 시트가 "4307-01"을 날짜로 바꾸지 않도록 붙이는 접두어
NO_VALUE = "-"  # 맥락이 해당 없는 행(off)의 표시
WIKIDOCS_PAGE = re.compile(r"wikidocs\.net/(\d+)")

OK_REASON = "없음"
REJECT_REASONS = [
    "정보 없음",
    "비유",
    "다른 언어",
    "추상적 주장",
    "개념 불일치",
]
SOURCE_PARAGRAPH, SOURCE_PAGE, SOURCE_NONE = "문단", "페이지", "없음"
# 결과 열 값
MADE, ALL_REJECTED, FAILED = "출제", "전부 거부", "오류"
NO_SENTENCE = "근거 제시 못함"  # off: LLM이 근거 문단을 주지 못함

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
OFF_PARAGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "source_url": {"type": "string"},
        "source_paragraph": {"type": "string"},
    },
    "required": ["question", "answer", "source_url", "source_paragraph"],
    "additionalProperties": False,
}
PARAGRAPH_SYSTEM = f"""너는 파이썬 학습 퀴즈 출제자야.
아래 [개념]을 확인하는 빈칸 채우기 문제의 문장을 [핵심 문단]에 근거해서 새로 써 줘.
- [핵심 문단]이 1순위 근거야. 문제 내용과 정답은 반드시 [핵심 문단]에 있는 내용이어야 하고, 문단 밖 지식은 쓰지 마.
- [같은 페이지]는 2순위 참고 자료야. 그 안의 {"<<핵심 문단>>"} 자리가 [핵심 문단]이야. 정의·예제 이름·앞뒤 코드를 이해하는 데만 쓰고, 정답이나 문제 내용을 여기에서만 가져오지 마. [같은 페이지]가 없으면 무시해.
- question: 학습자가 [핵심 문단]을 보지 않고도 이해할 수 있는 자기완결적인 한 문장. 문단의 문장을 그대로 베끼지 말고 직접 써. 문단 안에서 정의되지 않은 변수·값 이름은 쓰지 마. 접속어로 시작하지 마.
  정답이 들어갈 자리는 "{BLANK}"로 쓰고, 같은 단어가 여러 번 나오면 모두 "{BLANK}"로 가려.
- answer: {BLANK}에 들어갈 파이썬 개념 용어(조사 제외). [핵심 문단]에도 나와야 하고, question의 다른 곳에는 남아 있으면 안 돼.
- source: 문제 내용의 근거가 [핵심 문단]이면 "{SOURCE_PARAGRAPH}", [같은 페이지]에서만 나오면 "{SOURCE_PAGE}".
거부 사유 (문제를 낼 수 없을 때 하나 골라):
- 정보 없음: 문단이 도입·예고·안내·잡담·감정·책 이야기뿐
- 비유: 문단 전체가 일상 사물에 빗댄 설명
- 다른 언어: 파이썬이 아닌 언어 이야기
- 추상적 주장: 확인할 수 있는 사실이 없음
- 개념 불일치: 문단이 [개념]을 다루지 않음
출제하면 reject_reason은 "{OK_REASON}", 거부하면 question·answer는 빈 문자열, source는 "{SOURCE_NONE}".
JSON 객체 하나만 출력해."""
OFF_PARAGRAPH_SYSTEM = f"""너는 파이썬 학습 도우미야.
학습자가 아래 책에서 아래 개념을 공부했어. 책의 내용에 기반한 빈칸 채우기 문제의 문장 하나를 새로 쓰고, 그 근거가 된 책의 문단과 URL을 함께 써 줘.
- question: 개념을 확인하는 한 문장. 정답이 들어갈 자리는 "{BLANK}"로 쓰고, 같은 단어가 여러 번 나오면 모두 "{BLANK}"로 가려. answer: {BLANK}에 들어갈 파이썬 개념 용어(조사 제외).
- source_url: 근거 문단이 있는 책 페이지 URL. source_paragraph: 그 문단을 책에 적힌 그대로.
웹을 볼 수 없으니 기억나는 대로 써. 정확히 기억나지 않아도 책에 있을 법한 가장 가까운 내용을 써.
JSON 객체 하나만 출력해."""

FIELDS = [
    "id",
    "book_id",
    "성격",
    "rag",
    "맥락",  # on: 페이지 맥락 on/off (off는 -)
    "개념",
    "결과",  # 출제 / 전부 거부 / 근거 제시 못함 / 오류
    "문제",
    "답",
    "빈칸 수",
    "근거 문단",  # off는 LLM이 제시한 문단
    "근거 ID",
    "근거 소제목",
    "근거 URL",
    "근거 점수",  # on: 검색 점수(코사인 유사도)
    "문단 글자 수",
    "이미지 포함",  # on: 근거 문단에 이미지가 있었나 (표시는 프롬프트에서 뺌) o/x
    "그림 참조",  # on: 근거 문단에 "그림 N"이 있나 o/x. 내용이 그림에 있을 수 있다
    "맥락 글자 수",
    "출제 순위",  # on: 몇 번째 순위에서 출제됐나
    "근거 후보 수",  # on: 검색이 돌려준 후보 수
    "거부 이력",  # on: "순위:사유" 목록
    "근거 위치",  # on: LLM이 말한 문제 내용의 근거 (문단/페이지)
    "원문 일치",  # 새 문장을 채운 것이 문단에 그대로 들어 있나 o/x
    "호출 수",
    "입력 글자 수",  # 호출한 프롬프트(시스템+사용자) 글자 수의 합. 토큰 비용의 대용치
    "URL 책 존재",  # off: 근거 URL의 페이지가 그 책에 있나 o/x
    "문단 유사도",  # off: 제시 문단과 그 페이지 문단의 최대 코사인 유사도
    "제시 문단 속 답",  # off: 답이 제시한 문단에 있나 o/x
    "근거_sim",  # on: 문제(빈칸 문장)와 근거 문단 유사도. 임베딩 점검용
    "근거 확인(o/x)",
    "답 유일(o/x)",
    "핵심 개념(o/x)",
    "거부 타당(o/x)",
    "오류",
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


def label(item_id: str) -> str:
    return f"{ID_PREFIX}{item_id}"


def unlabel(value: str) -> str:
    return value.removeprefix(ID_PREFIX)


# ---------------------------------------------------------------- 빈칸


class BlankError(ValueError):
    """LLM이 쓴 문장·정답으로 빈칸 문제를 만들 수 없음."""


def normalize(text: str) -> str:
    """공백·굽은 따옴표·끝 문장부호 차이를 없앤 비교용 문자열."""
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    return re.sub(r"\s+", "", text).rstrip(".!?")


def contains(text: str, answer: str) -> bool:
    """정답이 텍스트에 들어 있나 (공백·굽은 따옴표 차이는 무시)."""
    return normalize(answer) in normalize(text)


def answer_pattern(answer: str) -> re.Pattern:
    """문장 안에서 정답과 같은 단어를 찾는 정규식. 영문 식별자의 일부(print 안의 int 등)는 제외."""
    left = r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", answer[0]) else ""
    right = r"(?![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", answer[-1]) else ""
    return re.compile(f"{left}{re.escape(answer)}{right}")


def make_question(text: str, answer: str) -> tuple[str, int]:
    """LLM이 쓴 문장으로 (빈칸 문제, 빈칸 수). 이미 ____로 비워 왔으면 그대로 쓰되 남은 정답도 가린다.

    ____가 없으면 문장에서 정답을 찾아 직접 가린다 (blank_out).
    """
    answer = answer.strip()
    if BLANK not in text:
        return blank_out(text, answer)
    if not answer:
        raise BlankError("정답이 비어 있음")
    if len(answer) > MAX_ANSWER_CHARS:
        raise BlankError(f"정답이 너무 김({len(answer)}자)")
    question = answer_pattern(answer).sub(BLANK, text)  # 같은 단어가 남아 있으면 가린다
    if question.replace(BLANK, "").strip() == "":
        raise BlankError("문제에 빈칸 말고 내용이 없음")
    return question, question.count(BLANK)


def blank_out(sentence: str, answer: str) -> tuple[str, int]:
    """문장에서 정답을 모두 가린 (문제, 빈칸 수). 가릴 수 없으면 BlankError."""
    answer = answer.strip()
    if not answer:
        raise BlankError("정답이 비어 있음")
    if len(answer) > MAX_ANSWER_CHARS:
        raise BlankError(f"정답이 너무 김({len(answer)}자)")
    if BLANK in sentence:
        raise BlankError("원문에 빈칸 표시가 이미 있음")
    if answer not in sentence:
        raise BlankError(f"정답이 문장에 없음: {answer}")
    if normalize(answer) == normalize(sentence):
        raise BlankError("정답이 문장 전체")

    question, count = answer_pattern(answer).subn(BLANK, sentence)
    if count == 0:  # 같은 단어가 없으면 있는 그대로 첫 번째 하나만 가린다
        question, count = sentence.replace(answer, BLANK, 1), 1
    if normalize(question.replace(BLANK, answer)) != normalize(sentence):
        raise BlankError("채운 문장이 원문과 다름")
    return question, count


def filled(question: str, answer: str) -> str:
    """빈칸을 정답으로 채운 문장."""
    return question.replace(BLANK, answer)


# ---------------------------------------------------------------- LLM


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
        if "per day" in str(e):
            raise DailyLimitReached(str(e)) from e
        raise
    match = qs.JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"JSON 없음: {text[:120]}")
    return json.loads(match.group())  # JSONDecodeError는 ValueError의 하위 클래스


def paragraph_prompt(concept: str, unit: dict, context: str | None) -> str:
    user = f"[개념]\n{concept}\n\n[핵심 문단]\n{unit['text']}"
    if context is not None:
        user += f"\n\n[같은 페이지]\n{context}"
    return user


def ask_paragraph(client, model: str, user: str) -> dict:
    """문단 방식 출제. 형식이 어긋나면 ValueError."""
    data = call_json(client, model, PARAGRAPH_SYSTEM, user, PARAGRAPH_SCHEMA)
    out = {
        "reject_reason": str(data.get("reject_reason", "")),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "source": str(data.get("source", "")),
    }
    if out["reject_reason"] != OK_REASON and out["reject_reason"] not in REJECT_REASONS:
        raise ValueError(f"알 수 없는 거부 사유: {out['reject_reason']!r}")
    return out


def validate_paragraph(out: dict, unit_text: str) -> tuple[str, int]:
    """출제 결과를 코드로 검증해 (문제, 빈칸 수). 실패하면 BlankError.

    LLM이 말한 근거 위치(source)는 기록만 하고 거부 기준으로 쓰지 않는다. 정답이 핵심 문단에
    있는지를 코드가 직접 확인하는 쪽이 더 믿을 만하다.
    """
    question, count = make_question(out["question"], out["answer"])
    if not contains(unit_text, out["answer"]):
        raise BlankError("정답이 핵심 문단에 없음")
    return question, count


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


def run_on(
    client, model, pi: ParagraphIndex, base: dict, used: set[str], tries: int
) -> dict:
    book_id, concept = base["book_id"], base["개념"]
    with_context = base["맥락"] == "on"
    found = pi.find(concept, book_id, frozenset(used))
    row = {**base, "근거 후보 수": len(found), "호출 수": 0, "입력 글자 수": 0}
    if not found:
        return {**row, "결과": FAILED, "오류": "근거 문단 없음"}
    rejects: list[str] = []
    for rank, unit in enumerate(found[:tries], start=1):
        context = pi.page_context(unit) if with_context else None
        user = paragraph_prompt(concept, unit, context)
        out, error = None, None
        # 생성·형식 오류(json_validate_failed 등)는 같은 문단으로 다시 시도한다.
        # 그래도 안 되면 그 순위만 실패로 기록하고 넘어간다
        for attempt in range(1, 1 + CALL_RETRIES + 1):
            row["호출 수"] += 1
            row["입력 글자 수"] += len(PARAGRAPH_SYSTEM) + len(user)
            try:
                out = ask_paragraph(client, model, user)
                break
            except (ValueError, APIError) as e:
                error = e
                logger.warning(
                    "[%s] %d위 호출 실패 (%d/%d): %s",
                    base["id"], rank, attempt, 1 + CALL_RETRIES, e,
                )  # fmt: skip
        if out is None:
            out = {"reject_reason": f"호출 실패({type(error).__name__})"}
        reason = out["reject_reason"]
        if reason == OK_REASON:
            try:
                question, count = validate_paragraph(out, unit["text"])
            except BlankError as e:
                reason = f"검증 실패({e})"
        if reason != OK_REASON:
            rejects.append(f"{rank}:{reason}")
            logger.info(
                "[%s/on/%s] %d위 거부 %s: %s",
                base["id"], base["맥락"], rank, reason,
                " ".join(unit["text"].split())[:60],
            )  # fmt: skip
            continue
        used.add(unit["unit_id"])
        sim = float(np.dot(*embed(pi.model, [question, unit["embed_text"]])))
        return {
            **row,
            "결과": MADE,
            "문제": question,
            "답": out["answer"],
            "빈칸 수": count,
            "근거 문단": unit["text"],
            "근거 ID": label(unit["unit_id"]),
            "근거 소제목": unit["section"],
            "근거 URL": unit["url"],
            "근거 점수": unit["score"],
            "문단 글자 수": len(unit["text"]),
            "이미지 포함": "o" if unit["has_image"] else "x",
            "그림 참조": "o" if unit["figure_ref"] else "x",
            "맥락 글자 수": len(context) if context is not None else "",
            "출제 순위": rank,
            "거부 이력": " / ".join(rejects),
            "근거 위치": out["source"],
            "원문 일치": "o"
            if contains(unit["text"], filled(question, out["answer"]))
            else "x",
            "근거_sim": f"{sim:.4f}",
        }
    return {**row, "결과": ALL_REJECTED, "거부 이력": " / ".join(rejects)}


def page_ids(chunk_index: dict[str, dict], book_id: int) -> dict[int, list[str]]:
    """책의 page_id → 그 페이지의 chunk_id 목록."""
    pages: dict[int, list[str]] = {}
    for chunk in chunk_index.values():
        if chunk.get("book_id") == book_id:
            pages.setdefault(chunk["page_id"], []).append(chunk["chunk_id"])
    return pages


def run_off(client, model, pi: ParagraphIndex, base: dict, book: str) -> dict:
    book_id, concept = base["book_id"], base["개념"]
    user = f"[책]\n{book} ({BOOK_URL.format(book_id=book_id)})\n\n[개념]\n{concept}"
    data = call_json(client, model, OFF_PARAGRAPH_SYSTEM, user, OFF_PARAGRAPH_SCHEMA)
    row = {
        **base,
        "호출 수": 1,
        "입력 글자 수": len(OFF_PARAGRAPH_SYSTEM) + len(user),
    }
    text = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    url = str(data.get("source_url", "")).strip()
    paragraph = str(data.get("source_paragraph", "")).strip()
    if not (text and answer and paragraph):
        return {**row, "결과": NO_SENTENCE}

    row |= {"근거 문단": paragraph, "근거 URL": url}
    match = WIKIDOCS_PAGE.search(url)
    chunk_ids = (
        page_ids(pi.chunk_index, book_id).get(int(match[1]), []) if match else []
    )
    row["URL 책 존재"] = "o" if chunk_ids else "x"
    if chunk_ids:
        unit_ids = [
            uid for uid, r in pi.rows(book_id).items() if r["chunk_id"] in chunk_ids
        ]
        if unit_ids:
            vec = np.asarray(embed(pi.model, [paragraph])[0], dtype=np.float32)
            row["문단 유사도"] = f"{float((pi.vectors(unit_ids) @ vec).max()):.4f}"
    row["제시 문단 속 답"] = "o" if contains(paragraph, answer) else "x"
    try:
        question, count = make_question(text, answer)
    except BlankError as e:
        return {**row, "결과": FAILED, "오류": f"검증 실패({e})"}
    return {
        **row,
        "결과": MADE,
        "문제": question,
        "답": answer,
        "빈칸 수": count,
        "원문 일치": "o" if contains(paragraph, filled(question, answer)) else "x",
    }


# ---------------------------------------------------------------- 이어 실행


def row_key(row: dict) -> tuple[str, str, str]:
    return row["id"], row["rag"], row["맥락"]


def load_finished(path: Path) -> list[dict]:
    """이전 결과에서 끝난 행(출제·전부 거부·근거 제시 못함)만. 오류 행은 다시 실행하도록 버린다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"--resume 파일의 열이 현재 형식과 다릅니다: {sorted(missing)}"
            )
        return [r for r in reader if r.get("결과") in (MADE, ALL_REJECTED, NO_SENTENCE)]


# ---------------------------------------------------------------- 요약


def summarize_group(title: str, on: list[dict]) -> None:
    counts = Counter(r["결과"] for r in on)
    print(f"\n--- {title} (n={len(on)}) ---")
    print("  " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    ranks = Counter(int(r["출제 순위"]) for r in on if r.get("출제 순위"))
    if ranks:
        print("  출제 순위: " + "  ".join(f"{k}위 {ranks[k]}" for k in sorted(ranks)))
    attempts = Counter()
    for r in on:
        for item in filter(None, (r.get("거부 이력") or "").split(" / ")):
            attempts[item.split(":", 1)[1].split("(")[0]] += 1
    n_try = sum(attempts.values()) + sum(r["결과"] == MADE for r in on)
    if attempts:
        print(f"  거부 사유 (전체 시도 {n_try}번 중):")
        for reason, n in attempts.most_common():
            print(f"    {reason}: {n} ({n / n_try:.0%})")
    calls = [int(r["호출 수"]) for r in on if r.get("호출 수")]
    chars = [int(r["입력 글자 수"]) for r in on if r.get("입력 글자 수")]
    if calls:
        print(
            f"  호출 수 합 {sum(calls)}, 케이스당 평균 {statistics.mean(calls):.1f}, "
            f"입력 글자 수 평균 {statistics.mean(chars):.0f}/케이스"
        )
    plen = [int(r["문단 글자 수"]) for r in on if r.get("문단 글자 수")]
    if plen:
        print(
            f"  출제 문단 글자 수: 평균 {statistics.mean(plen):.0f} ({min(plen)}~{max(plen)})"
        )
    blanks = Counter(r["빈칸 수"] for r in on if r.get("빈칸 수"))
    if blanks:
        print(
            "  빈칸 수: " + "  ".join(f"{k}개 {v}" for k, v in sorted(blanks.items()))
        )
    sources = Counter(r["근거 위치"] for r in on if r.get("근거 위치"))
    if sources:
        print("  근거 위치: " + "  ".join(f"{k} {v}" for k, v in sources.items()))
    flagged = [r for r in on if r["결과"] == MADE]
    if flagged:
        img = sum(r["이미지 포함"] == "o" for r in flagged)
        fig = sum(r["그림 참조"] == "o" for r in flagged)
        print(f"  출제 문단 중 이미지 포함 {img}, 그림 참조 {fig}")
    sims = [float(r["근거_sim"]) for r in on if r.get("근거_sim")]
    if sims:
        print(f"  근거_sim 평균 {statistics.mean(sims):.4f} 최소 {min(sims):.4f}")


def summarize(rows: list[dict]) -> None:
    print("\n=== 결과 ===")
    contexts = dict.fromkeys(r["맥락"] for r in rows if r["rag"] == "on")
    for ctx in contexts:
        summarize_group(
            f"on, 페이지 맥락 {ctx}",
            [r for r in rows if r["rag"] == "on" and r["맥락"] == ctx],
        )
    off = [r for r in rows if r["rag"] == "off"]
    if off:
        counts = Counter(r["결과"] for r in off)
        print(f"\n--- off (n={len(off)}) ---")
        print("  " + "  ".join(f"{k} {v}" for k, v in counts.items()))
        given = [r for r in off if r["결과"] != NO_SENTENCE]
        if given:
            url_ok = sum(r.get("URL 책 존재") == "o" for r in given)
            sims = [float(r["문단 유사도"]) for r in given if r.get("문단 유사도")]
            print(f"  근거 URL의 페이지가 책에 있음: {url_ok}/{len(given)}")
            if sims:
                print(
                    f"  제시 문단과 그 페이지 문단 유사도: 평균 {statistics.mean(sims):.3f}"
                    f" 최대 {max(sims):.3f} (n={len(sims)})"
                )


# ---------------------------------------------------------------- 실행


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG on/off 빈칸 문제 정확도 테스트")
    parser.add_argument("--cases", type=Path, default=CASES_CSV)
    parser.add_argument("--ids", nargs="+", help="실행할 케이스 id (생략하면 전체)")
    parser.add_argument(
        "--context",
        nargs="+",
        choices=["on", "off"],
        default=["off"],
        help="[같은 페이지] 맥락을 줄지. 여러 개면 모두 비교 (기본: off)",
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
        help="on에서 거부(또는 코드 검증 실패)될 때 시도할 최대 순위",
    )
    parser.add_argument(
        "--model", default=DEFAULT_EVAL_MODEL, help="문제 생성 LLM (Groq 모델 ID)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="이전 결과 CSV. 끝난 (케이스, rag, 맥락)은 건너뛴다",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="LLM 없이 검색과 book_id 필터 검사만"
    )
    args = parser.parse_args()
    llm_model = args.model
    contexts = list(dict.fromkeys(args.context))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cases = load_cases(args.cases)
    if args.ids:
        cases = [c for c in cases if c["id"].strip() in set(args.ids)]
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

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=GROQ_MAX_RETRIES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if args.resume:
        # 새 파일을 열기 전에 먼저 읽는다. 이전 파일은 그대로 두고 끝난 행만 옮긴다
        rows = load_finished(args.resume)
        logger.info("이어서 실행: 끝난 %d행을 가져옴 (%s)", len(rows), args.resume)
    done = {row_key(r) for r in rows}
    used: dict[tuple[str, int], set[str]] = {}
    for r in rows:
        if r["rag"] == "on" and r.get("근거 ID"):
            key = (r["맥락"], int(r["book_id"]))
            used.setdefault(key, set()).add(unlabel(r["근거 ID"]))

    out = (
        OUT_DIR
        / f"blank_{stamp}_{llm_model.split('/')[-1]}_ctx{'-'.join(contexts)}.csv"
    )
    if args.resume and out.resolve() == args.resume.resolve():
        raise SystemExit(f"결과 파일이 --resume 파일과 같습니다: {out}")
    stop_reason: DailyLimitReached | None = None
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
            # (rag, 맥락) 실행 목록: on은 맥락마다, off는 케이스당 한 번
            jobs = [("on", ctx) for ctx in contexts]
            if not args.skip_off:
                jobs.append(("off", NO_VALUE))
            for rag, ctx in jobs:
                base = {
                    "id": cid,
                    "book_id": book_id,
                    "성격": case.get("성격", ""),
                    "rag": rag,
                    "맥락": ctx,
                    "개념": concept,
                }
                if row_key(base) in done:
                    continue
                tag = f"{cid}/{rag}/{ctx}"
                try:
                    if rag == "on":
                        used_set = used.setdefault((ctx, book_id), set())
                        row = run_on(
                            client, llm_model, pi, base, used_set, args.max_tries
                        )
                    else:
                        row = run_off(client, llm_model, pi, base, book)
                except DailyLimitReached as e:
                    stop_reason = e
                    break  # 이 작업은 저장하지 않는다 → --resume 때 다시 만든다
                except (ValueError, APIError) as e:
                    row = {**base, "결과": FAILED, "오류": f"{type(e).__name__}: {e}"}
                    logger.warning("[%s] 실패: %s", tag, row["오류"])
                logger.info("[%s] %s %s", tag, row["결과"], row.get("문제", ""))
                rows.append(row)
                writer.writerow(row)
                f.flush()
            if stop_reason:
                break

    if stop_reason:
        logger.error("하루 한도에 걸려 중단했습니다: %s", stop_reason)
        print(
            f"\n하루 한도로 중단. 한도가 풀리면 이어서 실행하세요:\n"
            f"  python eval_blank.py --resume {out} --model {llm_model} "
            f"--context {' '.join(contexts)}"
            + (" --skip-off" if args.skip_off else "")
            + (f" --ids {' '.join(args.ids)}" if args.ids else "")
        )
    summarize(rows)
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
