"""임베딩한 개념 조각을 근거로 질문에 답하고, 확인 퀴즈를 만든다.

검색(text_embed.search) → 조각 본문 조립 → Groq LLM 답변 → 퀴즈 생성 순서로 동작한다.
답변과 퀴즈는 검색된 조각에만 근거하며, 출처 번호와 URL을 함께 보여 준다.

.env:
    GROQ_API_KEY=...
    GROQ_MODEL=qwen/qwen3.8-27b   # 선택. 생략하면 기본값

사용 예:
    python text_answer.py "문자열 공백 제거는 어떻게 해?"
    python text_answer.py "리스트 슬라이싱" --book-id 1 --k 4
    python text_answer.py "딕셔너리" --quiz 5 --solve   # 퀴즈를 터미널에서 직접 풀기
    python text_answer.py "튜플" --no-quiz
"""

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from text_embed import DATA_ROOT, get_collection, load_chunks, load_model, search

DEFAULT_MODEL = "qwen/qwen3.8-27b"
ANSWER_TEMPERATURE = 0.2
QUIZ_TEMPERATURE = 0.5
QUIZ_EXTRA = 2  # 겹치지 않는 문제를 고를 수 있도록 후보를 더 만든다
MAX_SOURCE_CHARS = 4000  # 조각 하나가 컨텍스트를 독차지하지 않도록 자른다
# 관련 없는 조각이 근거로 섞이지 않도록 거르는 기준 (코사인 거리, 작을수록 가깝다)
MAX_DISTANCE = 0.5  # 이보다 먼 조각은 버린다
DISTANCE_MARGIN = 0.06  # 1위보다 이만큼 이상 먼 조각은 버린다
THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
CITATION = re.compile(r"\[(\d+)\]")
# 해설이 보기 번호를 가리키면 섞었을 때 틀린 해설이 된다
CHOICE_REFERENCE = re.compile(r"(선택지|보기)\s*\d|\d\s*번(?!째)")

logger = logging.getLogger("text_answer")

ANSWER_SYSTEM = """너는 프로그래밍 학습을 돕는 튜터야.
- 반드시 아래 [자료]에 있는 내용만 근거로 한국어로 답해.
- 문장 끝에 근거가 된 자료 번호를 [1], [2]처럼 붙여.
- 자료에 답이 없으면 추측하지 말고 "제공된 자료에서 찾을 수 없어요."라고만 답해.
- 필요하면 자료의 코드 예시를 짧게 인용해."""

SKILLS = ("개념 이해", "결과 예측", "함수 선택", "오류 찾기")

QUIZ_SYSTEM = """너는 학습 직후 기억을 확인하는 퀴즈 출제자야.
- [자료]에 있는 내용만으로 풀 수 있는 문제를 만들어. 자료 밖 지식은 쓰지 마.
- 문제마다 묻는 개념(concept)과 유형(skill)을 다르게 해.
  같은 유형에서 함수 이름이나 값만 바꾼 비슷한 문제는 만들지 마.
- skill은 다음 중 하나:
  "개념 이해": 동작 원리나 차이를 바르게 설명한 것 고르기
  "결과 예측": 코드 실행 결과 맞히기
  "함수 선택": 주어진 상황에 맞는 함수·문법 고르기
  "오류 찾기": 틀린 설명, 잘못된 코드, 오류가 나는 경우 찾기
- concept는 묻는 대상을 짧게 (예: "lstrip", "문자열 슬라이싱").
- 객관식(mcq)은 서로 다른 보기 4개, 정답 1개.
  오답은 학습자가 실제로 헷갈리는 것으로 만들어:
  비슷한 함수와의 혼동, 방향·범위 착각, 흔한 오해.
  말이 안 되는 보기, 정답만 유난히 길거나 자세한 보기는 금지.
- 보기나 정답이 문자열 값이면 'hi '처럼 따옴표로 감싸서 앞뒤 공백이 보이게 써.
- 단답형(short)은 한 단어나 한 줄 코드로 답할 수 있게 만들어.
- 해설에는 정답의 근거와, 가장 헷갈리는 오답이 왜 틀렸는지를 함께 써.
  보기 순서는 나중에 섞이므로 해설에서 "1번", "선택지 2" 같은 번호를 쓰지 말고
  보기 내용으로 가리켜.
- 출력은 아래 형식의 JSON 객체 하나만. 다른 텍스트는 쓰지 마.
{"questions": [
  {"type": "mcq", "skill": "결과 예측", "concept": "...", "question": "...",
   "choices": ["...", "...", "...", "..."], "answer_index": 0,
   "explanation": "...", "source": 1},
  {"type": "short", "skill": "함수 선택", "concept": "...", "question": "...",
   "answer": "...", "explanation": "...", "source": 2}
]}"""


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


def chat(client, messages: list[dict], temperature: float) -> str:
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    # 추론 모델이 생각 과정을 본문에 섞어 내보내는 경우를 걸러낸다
    return THINK_TAG.sub("", content).strip()


def answer(client, question: str, sources: list[Source]) -> str:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"[자료]\n{format_sources(sources)}\n\n[질문]\n{question}",
        },
    ]
    return chat(client, messages, ANSWER_TEMPERATURE)


def cited_sources(answer_text: str, sources: list[Source]) -> list[Source]:
    """답변이 실제로 인용한 근거만 남긴다. 인용이 없으면 빈 목록."""
    cited = {int(n) for n in CITATION.findall(answer_text)}
    return [s for s in sources if s.no in cited]


def quiz_problem(q: dict) -> str | None:
    """문제 형식의 결함. 정상이면 None."""
    if q.get("type") != "mcq":
        return None if q.get("answer") else "정답 없음"
    choices = q.get("choices")
    idx = q.get("answer_index")
    if not isinstance(choices, list) or len(choices) < 2:
        return "보기 부족"
    # 공백 문제처럼 앞뒤 공백만 다른 보기가 정상이므로 원문 그대로 비교한다
    if len(set(map(str, choices))) != len(choices):
        return "중복 보기"
    if not isinstance(idx, int) or not 0 <= idx < len(choices):
        return f"answer_index 범위 밖 ({idx})"
    return None


def parse_quiz(text: str, source_nos: set[int]) -> list[dict]:
    match = JSON_OBJECT.search(text)
    if not match:
        raise QuizFormatError("JSON 객체를 찾지 못함")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise QuizFormatError(f"JSON 파싱 실패: {e}") from e

    questions = []
    for q in data.get("questions", []):
        if not isinstance(q, dict) or not q.get("question"):
            continue
        if q.get("type") != "mcq":
            q["type"] = "short"
        problem = quiz_problem(q)
        if problem:
            logger.warning("문제 제외 (%s): %s", problem, q["question"])
            continue
        if q.get("source") not in source_nos:
            q["source"] = None
        if q.get("skill") not in SKILLS:
            q["skill"] = ""
        # 개념이 비면 문제마다 다른 것으로 취급한다
        q["concept"] = str(q.get("concept") or q["question"]).strip().lower()
        questions.append(q)
    if not questions:
        raise QuizFormatError("유효한 문제가 없음")
    return questions


def shuffle_choices(q: dict, rng: random.Random) -> None:
    """모델은 정답을 앞쪽 보기에 두는 경향이 있어 순서를 섞는다.

    해설이 보기 번호를 가리키면 섞는 순간 해설이 틀리므로 그대로 둔다.
    """
    if CHOICE_REFERENCE.search(str(q.get("explanation", ""))):
        logger.info("해설이 보기 번호를 언급해 순서를 유지: %s", q["question"])
        return
    answer = q["choices"][q["answer_index"]]
    rng.shuffle(q["choices"])
    q["answer_index"] = q["choices"].index(answer)


def select_diverse(questions: list[dict], n: int) -> list[dict]:
    """유형과 개념이 겹치지 않는 문제부터 고르고, 모자라면 조건을 완화한다.

    같은 유형 반복이 더 단조롭게 느껴지므로 개념보다 유형 다양성을 먼저 지킨다.
    """
    picked: list[dict] = []
    skills: set[str] = set()
    concepts: set[str] = set()
    rules = [
        lambda q: q["skill"] not in skills and q["concept"] not in concepts,
        lambda q: q["skill"] not in skills,
        lambda q: q["concept"] not in concepts,
        lambda q: True,
    ]
    for rule in rules:
        for q in questions:
            if len(picked) == n:
                return picked
            if any(q is p for p in picked) or not rule(q):
                continue
            picked.append(q)
            skills.add(q["skill"])
            concepts.add(q["concept"])
    return picked


def make_quiz(
    client,
    question: str,
    sources: list[Source],
    n: int,
    retries: int = 1,
    rng: random.Random | None = None,
) -> list[dict]:
    messages = [
        {"role": "system", "content": QUIZ_SYSTEM},
        {
            "role": "user",
            "content": (
                f"[자료]\n{format_sources(sources)}\n\n"
                f"[학습자가 방금 공부한 질문]\n{question}\n\n"
                f"이 내용을 확인하는 후보 문제를 {n + QUIZ_EXTRA}개 만들어. "
                "객관식과 단답형을 섞고, 가능한 한 서로 다른 skill을 써."
            ),
        },
    ]
    rng = rng or random.Random()
    for attempt in range(retries + 1):
        text = chat(client, messages, QUIZ_TEMPERATURE)
        try:
            candidates = parse_quiz(text, {s.no for s in sources})
        except QuizFormatError as e:
            logger.warning("퀴즈 형식 오류 (%d회차): %s", attempt + 1, e)
            continue
        picked = select_diverse(candidates, n)
        logger.info(
            "퀴즈 후보 %d개 중 %d개 선택: %s",
            len(candidates),
            len(picked),
            [f"{q['skill'] or '-'}/{q['concept']}" for q in picked],
        )
        for q in picked:
            if q["type"] == "mcq":
                shuffle_choices(q, rng)
        return picked
    raise QuizFormatError("퀴즈 생성 실패")


# ---------------------------------------------------------------- 출력·풀이


def print_quiz(questions: list[dict], sources: list[Source], solve: bool) -> None:
    by_no = {s.no: s for s in sources}
    correct = 0
    for i, q in enumerate(questions, start=1):
        tag = f"[{q['skill']}] " if q["skill"] else ""
        print(f"\nQ{i}. {tag}{q['question']}")
        if q["type"] == "mcq":
            for n, choice in enumerate(q["choices"], start=1):
                print(f"   {n}) {choice}")
            answer_text = f"{q['answer_index'] + 1}) {q['choices'][q['answer_index']]}"
        else:
            answer_text = q["answer"]

        if solve:
            reply = input("   답: ").strip()
            # 메타인지 점검: 정답을 보기 전에 스스로 확신도를 매긴다
            confidence = input("   확신도 (1 모름 / 2 애매 / 3 확실): ").strip()
            if q["type"] == "mcq":
                ok = reply == str(q["answer_index"] + 1)
                correct += ok
                print(f"   → {'정답' if ok else '오답'} (확신도 {confidence or '-'})")
            else:
                print(f"   → 확신도 {confidence or '-'}, 아래 정답과 비교해 보세요")

        print(f"   정답: {answer_text}")
        print(f"   해설: {q.get('explanation', '')}")
        if q.get("source"):
            print(f"   근거: [{q['source']}] {by_no[q['source']].url}")

    n_mcq = sum(q["type"] == "mcq" for q in questions)
    if solve and n_mcq:
        print(f"\n객관식 {n_mcq}문제 중 {correct}문제 정답")


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
    parser.add_argument("--quiz", type=int, default=3, help="퀴즈 문제 수")
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
            questions = make_quiz(client, args.question, quiz_sources, args.quiz)
            print_quiz(questions, quiz_sources, args.solve)
    except APIConnectionError as e:
        logger.error("Groq 연결 실패: %s", e)
    except APIStatusError as e:
        logger.error("Groq API 오류 (%s): %s", e.status_code, e.message)
    except QuizFormatError as e:
        logger.error("%s", e)


if __name__ == "__main__":
    main()
