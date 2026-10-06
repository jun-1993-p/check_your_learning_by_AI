"""임베딩한 개념 조각을 근거로 질문에 답하고, 확인 퀴즈를 만든다.

검색(text_embed.search) → 조각 본문 조립 → Groq LLM 답변 → 퀴즈 생성 순서로 동작한다.
답변과 퀴즈는 검색된 조각에만 근거하며, 출처 번호와 URL을 함께 보여 준다.
퀴즈 유형별 스키마·검증·채점은 quiz_templates.py에 있다.

.env:
    GROQ_API_KEY=...
    GROQ_MODEL=qwen/qwen3.8-27b   # 선택. 생략하면 기본값

사용 예:
    python text_answer.py "문자열 공백 제거는 어떻게 해?"
    python text_answer.py "리스트 슬라이싱" --book-id 1 --k 4
    python text_answer.py "딕셔너리" --quiz 6 --solve        # 퀴즈를 터미널에서 직접 풀기
    python text_answer.py "문자열 공백 제거" --formats ox,blank --quiz 4
    python text_answer.py "튜플" --no-quiz
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
from text_embed import DATA_ROOT, get_collection, load_chunks, load_model, search

DEFAULT_MODEL = "qwen/qwen3.8-27b"
ANSWER_TEMPERATURE = 0.2
QUIZ_TEMPERATURE = 0.5
# Groq 무료 등급은 분당 출력 토큰이 1,000개라, 요청마다 상한을 명시해야 거절되지 않는다
ANSWER_MAX_TOKENS = 700
QUIZ_MAX_TOKENS = 900
MAX_SOURCE_CHARS = 4000  # 조각 하나가 컨텍스트를 독차지하지 않도록 자른다
# 관련 없는 조각이 근거로 섞이지 않도록 거르는 기준 (코사인 거리, 작을수록 가깝다)
MAX_DISTANCE = 0.5  # 이보다 먼 조각은 버린다
DISTANCE_MARGIN = 0.06  # 1위보다 이만큼 이상 먼 조각은 버린다
THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
CITATION = re.compile(r"\[(\d+)\]")

logger = logging.getLogger("text_answer")

ANSWER_SYSTEM = """너는 프로그래밍 학습을 돕는 튜터야.
- 반드시 아래 [자료]에 있는 내용만 근거로 한국어로 답해.
- 문장 끝에 근거가 된 자료 번호를 [1], [2]처럼 붙여.
- 자료에 답이 없으면 추측하지 말고 "제공된 자료에서 찾을 수 없어요."라고만 답해.
- 필요하면 자료의 코드 예시를 짧게 인용해."""

QUIZ_SYSTEM = """너는 학습 직후 기억을 확인하는 퀴즈 출제자야.
- 학습자는 방금 [튜터 답변]을 읽었어. 문제는 그 답변에서 다룬 내용에서만 내.
  [자료]에 있어도 답변에서 다루지 않은 내용(같은 절의 다른 함수 등)은 묻지 마.
- [자료]는 정답의 근거를 확인하는 데 써. 자료 밖 지식은 쓰지 마.
- 문제마다 묻는 개념(concept)을 다르게 하고, 값만 바꾼 비슷한 문제는 만들지 마.
- skill은 다음 중 하나:
  "개념 이해": 동작 원리나 차이를 이해했는지
  "결과 예측": 코드 실행 결과를 맞히는지
  "함수 선택": 상황에 맞는 함수·문법을 고르는지
  "오류 찾기": 틀린 설명, 잘못된 코드, 오류가 나는 경우를 찾는지
- concept는 묻는 대상을 짧게 (예: "lstrip", "문자열 슬라이싱").
- source는 근거가 된 자료 번호.
- 문자열 값은 'hi '처럼 따옴표로 감싸서 앞뒤 공백이 보이게 써.
- 문제에 코드가 필요하면 code에 >>> 형식으로 쓰고, 필요 없으면 빈 문자열로 둬.
- 해설은 짧게 핵심만.
유형별 규칙:
{formats}"""


@dataclass
class Source:
    """LLM에 넘기는 근거 하나. no는 프롬프트 안의 [n] 번호."""

    no: int
    chunk_id: str
    path: str  # 매칭된 위치까지의 경로 (소제목 매칭이면 그 소제목으로 끝난다)
    url: str
    distance: float
    text: str


class QuizFormatError(Exception):
    """모델 출력이 퀴즈 JSON 형식이 아니다."""


# ---------------------------------------------------------------- 근거 조립


def render_block(block: dict) -> str:
    """임베딩용과 달리 코드·오류 출력까지 그대로 살린다."""
    kind = block["type"]
    if kind in ("code", "code_example"):
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


def load_chunk_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for chunks_file in sorted(DATA_ROOT.glob("*/chunks.jsonl")):
        for chunk in load_chunks(chunks_file):
            if chunk["chunk_id"] in index:
                logger.warning("chunk_id 중복: %s (%s)", chunk["chunk_id"], chunks_file)
            index[chunk["chunk_id"]] = chunk
    return index


def locate_section(chunk: dict, hit: dict) -> tuple[str, str]:
    """매칭된 위치의 (경로, URL). 소제목 벡터로 매칭됐으면 그 소제목을 가리킨다.

    조각 path는 [책 경로, 첫 h2, 첫 h3]라서, 소제목 매칭일 때는
    책 경로 뒤를 그 소제목(h3면 실제로 속한 h2 포함)으로 바꾼다.
    """
    if hit["matched"] != "child":
        return hit["path"], hit["source_url"]
    headings = [b for b in chunk["blocks"] if b["type"] == "heading"]
    target = next(
        (
            n
            for n, b in enumerate(headings)
            if b["level"] in (2, 3) and b["text"] == hit["heading"]
        ),
        None,
    )
    if target is None:
        return hit["path"], hit["source_url"]

    # path 끝의 제목들(최대 h2·h3 두 개)을 떼어 책 경로만 남긴다.
    # 조각 블록에 h3가 있어도 path에는 없을 수 있어서, 블록 구조로 추측하지 않는다
    breadcrumb = list(chunk.get("path", []))
    texts = {b["text"] for b in headings}
    for _ in range(2):
        if breadcrumb and breadcrumb[-1] in texts:
            breadcrumb.pop()
    tail = [headings[target]["text"]]
    if headings[target]["level"] == 3:
        parent = next((b for b in reversed(headings[:target]) if b["level"] == 2), None)
        if parent:
            tail.insert(0, parent["text"])
    section_path = " > ".join([*breadcrumb, *tail])

    anchor = headings[target].get("anchor")
    base = chunk["source_url"].split("#")[0]
    return section_path, f"{base}#{anchor}" if anchor else hit["source_url"]


def filter_hits(hits: list[dict], max_distance: float, margin: float) -> list[dict]:
    if not hits:
        return []
    cutoff = min(max_distance, hits[0]["distance"] + margin)
    kept = [h for h in hits if h["distance"] <= cutoff]
    dropped = [f"{h['chunk_id']}({h['distance']:.3f})" for h in hits[len(kept) :]]
    if dropped:
        logger.info("거리 기준 %.3f 초과로 제외: %s", cutoff, ", ".join(dropped))
    return kept


def retrieve(
    question: str,
    k: int,
    book_id: int | None,
    model,
    collection,
    index: dict,
    max_distance: float = MAX_DISTANCE,
    margin: float = DISTANCE_MARGIN,
) -> list[Source]:
    hits = search(question, k, book_id, None, model=model, collection=collection)
    hits = filter_hits(hits, max_distance, margin)  # search는 거리순으로 돌려준다
    sources = []
    for hit in hits:
        chunk = index.get(hit["chunk_id"])
        if chunk is None:
            logger.warning(
                "chunks.jsonl에 없는 조각: %s (--rebuild 필요)", hit["chunk_id"]
            )
            continue
        body = "\n".join(t for b in chunk["blocks"] if (t := render_block(b).strip()))
        if len(body) > MAX_SOURCE_CHARS:
            body = body[:MAX_SOURCE_CHARS] + "\n...(생략)"
        path, url = locate_section(chunk, hit)
        sources.append(
            Source(
                no=len(sources) + 1,
                chunk_id=hit["chunk_id"],
                path=path,
                url=url,
                distance=hit["distance"],
                text=body,
            )
        )
    return sources


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


# ---------------------------------------------------------------- 퀴즈


def parse_llm_quiz(
    text: str, counts: dict[str, int], source_nos: set[int]
) -> dict[str, list[dict]]:
    match = JSON_OBJECT.search(text)
    if not match:
        raise QuizFormatError("JSON 객체를 찾지 못함")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise QuizFormatError(f"JSON 파싱 실패: {e}") from e

    result: dict[str, list[dict]] = {}
    for fmt, n in counts.items():
        items = data.get(fmt) if isinstance(data.get(fmt), list) else []
        kept = []
        for q in items:
            if not isinstance(q, dict):
                continue
            problem = qt.validate(fmt, q)
            if problem:
                label = q.get("question") or q.get("statement")
                logger.warning("문제 제외 (%s, %s): %s", fmt, problem, label)
                continue
            kept.append(qt.normalize(fmt, q, source_nos))
        if len(kept) < n:
            logger.warning("%s: 요청 %d개 중 %d개만 유효", fmt, n, len(kept))
        result[fmt] = kept[:n]
    if not any(result.values()):
        raise QuizFormatError("유효한 문제가 없음")
    return result


def failed_generation(error, limit: int = 400) -> str:
    """Groq가 스키마 검사에서 거절한 모델 출력의 앞부분. 실패 원인 파악용."""
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    text = str(detail.get("failed_generation", "")) if isinstance(detail, dict) else ""
    text = text.replace("\n", "⏎")
    return (text[:limit] + "…") if len(text) > limit else (text or "(없음)")


def llm_quiz(
    client,
    question: str,
    sources: list[Source],
    counts: dict[str, int],
    retries: int = 1,
    avoid_concepts: list[str] | None = None,
    answer_text: str = "",
) -> dict[str, list[dict]]:
    """avoid_concepts: 다른 유형(빈칸)에서 이미 출제한 개념. 같은 것을 다시 묻지 않게 한다.
    answer_text: 학습자가 방금 읽은 답변. 출제 범위를 이 내용으로 좁힌다.
    """
    request = "이 답변 내용을 확인하는 퀴즈를 만들어."
    if avoid_concepts:
        # "묻지 마"만 쓰면 주제 자체를 피해 무관한 내용으로 넘어간다
        request += (
            f" {', '.join(avoid_concepts)}은(는) 이미 다른 문제로 출제했으니, "
            "같은 주제 안에서 다른 개념이나 개념 사이의 차이를 물어."
        )
    messages = [
        {
            "role": "system",
            "content": QUIZ_SYSTEM.format(formats=qt.format_instructions(counts)),
        },
        {
            "role": "user",
            "content": (
                f"[자료]\n{format_sources(sources)}\n\n"
                f"[학습자가 방금 공부한 질문]\n{question}\n\n"
                f"[튜터 답변]\n{answer_text or '(없음)'}\n\n{request}"
            ),
        },
    ]
    from groq import BadRequestError

    schema = qt.build_schema(counts)
    for attempt in range(retries + 1):
        try:
            text = chat(client, messages, QUIZ_TEMPERATURE, QUIZ_MAX_TOKENS, schema)
            return parse_llm_quiz(text, counts, {s.no for s in sources})
        except QuizFormatError as e:
            logger.warning("퀴즈 형식 오류 (%d회차): %s", attempt + 1, e)
        except BadRequestError as e:
            # 모델 출력이 스키마 검사에 걸린 경우만 재시도한다. 그 밖의 400은 그대로 올린다
            if "json_validate_failed" not in str(e):
                raise
            logger.warning(
                "퀴즈 스키마 검사 실패 (%d회차), 다시 요청. 거절된 출력: %s",
                attempt + 1,
                failed_generation(e),
            )
    raise QuizFormatError("퀴즈 생성 실패")


def make_quiz(
    client,
    question: str,
    sources: list[Source],
    index: dict[str, dict],
    counts: dict[str, int],
    rng: random.Random | None = None,
    focus: str = "",
) -> list[dict]:
    """유형별로 문제를 만들어 --formats 순서대로 돌려준다.

    focus(답변 본문)는 빈칸 문제를 학습 주제와 관련된 예제부터 고르고,
    LLM 유형의 출제 범위를 좁히는 데 쓴다.
    """
    rng = rng or random.Random()
    by_format: dict[str, list[dict]] = {}

    if counts.get("blank"):
        chunks = [(s.no, index[s.chunk_id]) for s in sources]
        exclude = qt.defined_names(index.values())
        by_format["blank"] = qt.pick_blanks(
            chunks, counts["blank"], rng, focus, exclude
        )
        if len(by_format["blank"]) < counts["blank"]:
            logger.warning(
                "빈칸 재료 부족: 요청 %d개 중 %d개 (근거 조각에 쓸 만한 코드 예제가 적음)",
                counts["blank"],
                len(by_format["blank"]),
            )

    # 빈칸을 먼저 정해 두고, 그 개념을 LLM 유형에서 피하게 한다
    llm_counts = {f: n for f, n in counts.items() if f in qt.LLM_FORMATS}
    if llm_counts:
        blank_concepts = [q["concept"] for q in by_format.get("blank", [])]
        by_format.update(
            llm_quiz(
                client,
                question,
                sources,
                llm_counts,
                avoid_concepts=blank_concepts,
                answer_text=focus,
            )
        )
        # 한 유형이 통째로 빠지면(보기 중복 등) 그 유형만 한 번 더 요청한다.
        # 분당 토큰 한도 때문에 대기가 생길 수 있어 재요청은 한 번만 한다
        missing = {f: n for f, n in llm_counts.items() if not by_format.get(f)}
        if missing:
            logger.info("빠진 유형 다시 요청: %s", list(missing))
            done = list(
                dict.fromkeys(q["concept"] for qs in by_format.values() for q in qs)
            )
            try:
                by_format.update(
                    llm_quiz(
                        client,
                        question,
                        sources,
                        missing,
                        retries=0,
                        avoid_concepts=done,
                        answer_text=focus,
                    )
                )
            except QuizFormatError as e:
                logger.warning("재요청 실패, 빠진 유형 없이 진행: %s", e)
    for q in by_format.get("mcq", []):
        qt.shuffle_mcq(q, rng)

    quiz = [q for fmt in counts for q in by_format.get(fmt, [])]
    repeated = [c for c, n in Counter(q["concept"] for q in quiz).items() if n > 1]
    if repeated:
        logger.warning("여러 문제가 같은 개념을 묻습니다: %s", repeated)
    logger.info(
        "퀴즈 %d문제: %s",
        len(quiz),
        [f"{q['format']}/{q['skill'] or '-'}/{q['concept']}" for q in quiz],
    )
    return quiz


# ---------------------------------------------------------------- 출력·풀이


def print_quiz(questions: list[dict], sources: list[Source], solve: bool) -> None:
    by_no = {s.no: s for s in sources}
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
        if q.get("source"):
            print(f"   근거: [{q['source']}] {by_no[q['source']].url}")

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
    parser.add_argument("--k", type=int, default=5, help="근거 후보 조각 수 (최대)")
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
    parser.add_argument("--no-quiz", action="store_true")
    parser.add_argument("--solve", action="store_true", help="퀴즈를 직접 풀기")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from groq import APIConnectionError, APIStatusError

    client = get_client()
    index = load_chunk_index()
    sources = retrieve(
        args.question,
        args.k,
        args.book_id,
        load_model(),
        get_collection(),
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
                client, args.question, quiz_sources, index, counts, focus=answer_text
            )
            print_quiz(questions, quiz_sources, args.solve)
    except APIConnectionError as e:
        logger.error("Groq 연결 실패: %s", e)
    except APIStatusError as e:
        logger.error("Groq API 오류 (%s): %s", e.status_code, e.message)
    except QuizFormatError as e:
        logger.error("%s", e)


if __name__ == "__main__":
    main()
