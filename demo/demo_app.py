"""시연용 GUI: 학습(질문 → 답변·근거) → 확인 퀴즈(직접 풀기) → 정답과 해설(+근거 URL).

quiz_session.py의 검색·답변·출제 함수와 quiz_templates.py의 채점을 그대로 쓰고 화면만 얹는다.
- 질문 버튼 하나로 답변과 확인 문제(빈칸 1개, 시드 고정)까지 자동으로 만든다.
- 학습자가 답을 쓰고 제출하면 바로 채점하고 해설을 보여 준다. 정답 목록에 없는 답은
  LLM이 정답과 같은 뜻인지 판정한다 (evals/eval_blank.py의 동의 판정과 같은 방식).
- 정답·해설은 제출 전에는 화면에 없다.

실행 (프로젝트 루트에서. -m으로 실행해야 루트의 quiz_session 등을 import할 수 있다):
    PowerShell: $env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"; python -m streamlit run demo/demo_app.py

환경변수는 streamlit을 띄우기 전에 설정해야 한다. streamlit이 스크립트보다 먼저 protobuf를 불러와서,
text_paragraphs.get_collection()의 같은 우회 설정(conda-forge chromadb의 opentelemetry-proto 호환 문제)이
스크립트 안에서는 늦는다.
"""

import json
import logging
import random

import streamlit as st
from google.protobuf.internal import api_implementation

import quiz_session as qs
import quiz_templates as qt
from text_embed import load_chunk_index, load_model
from text_paragraphs import ParagraphIndex, get_collection

# 시연 질문 (잘 되는 질문 / 함정 질문 / 모르는 걸 모른다고 하는 질문). 녹화 전에 실제로 돌려 보고 고른다
EXAMPLE_QUESTIONS = [
    "문자열 공백 제거는 어떻게 해?",
    "튜플의 append 메서드는 어떻게 써?",  # 잘못된 전제: 튜플에는 append가 없다
    "도커 컨테이너는 어떻게 만들어?",
]
# 시연 설정은 고정한다: 빈칸 채우기 1문제, 같은 질문이면 같은 문제가 나오게 시드 고정
QUIZ_TOTAL = 1
QUIZ_FORMATS = ["blank"]
QUIZ_SEED = 0.5
STEPS = ["① 학습", "② 퀴즈", "③ 정답과 해설"]
MD_SPECIAL = "\\`*_{}[]<>()#+-.!|~$"

JUDGE_TEMPERATURE = 0.0
JUDGE_SYSTEM = """너는 빈칸 채우기 문제의 채점자야. [문제]의 빈칸에 [학습자 답]을 넣었을 때 [정답]과 같은 개념을 말하는지 판단해.
- 동의어, 한영 표기, 조사·접미 차이(strip / strip(), 변수 / variable, while / while문)는 맞는 것으로 봐.
- 더 넓거나 좁은 개념이거나 다른 대상이면 틀린 것으로 봐.
- reason: 판단 이유를 한국어 한 문장으로.
JSON 객체 하나만 출력해."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="학습 확인 퀴즈", page_icon="📘", layout="centered")


# ---------------------------------------------------------------- 준비·표시 도우미


@st.cache_resource(show_spinner="임베딩 모델과 문단 저장소를 불러오는 중…")
def load_resources():
    chunk_index = load_chunk_index()
    index = ParagraphIndex(load_model(), get_collection(), chunk_index)
    books = {
        c["book_id"]: c["book_title"]
        for c in chunk_index.values()
        if c.get("book_id") is not None and c.get("book_title")
    }
    return index, qs.get_client(), dict(sorted(books.items()))


def reset(keep_question: bool = False) -> None:
    """학습·퀴즈 상태를 지운다. 고른 책은 남기고, keep_question이면 질문창 내용도 남긴다."""
    keep = {"book_id", "question_input"} if keep_question else {"book_id"}
    for key in [k for k in st.session_state if k not in keep]:
        del st.session_state[key]


def md(text: str) -> str:
    """마크다운 특수 문자를 이스케이프하고 줄바꿈을 살린다 (문제의 ____가 굵게·가로줄로 바뀌지 않게)."""
    escaped = "".join(f"\\{ch}" if ch in MD_SPECIAL else ch for ch in str(text))
    return escaped.replace("\n", "  \n")


def highlight(text: str, phrase: str) -> str:
    """문단에서 근거 인용 구절을 굵게 표시한다. 그대로 찾을 수 없으면 강조 없이 둔다."""
    phrase = phrase.strip()
    pos = text.find(phrase) if phrase else -1
    if pos < 0:
        return md(text)
    end = pos + len(phrase)
    return f"{md(text[:pos])}**{md(phrase)}**{md(text[end:])}"


def stepper() -> None:
    state = st.session_state
    current = 2 if "record" in state else 1 if state.get("quiz") else 0
    for n, (col, label) in enumerate(zip(st.columns(len(STEPS)), STEPS)):
        color = "blue" if n == current else "gray"
        col.markdown(f"#### :{color}[{label}]")
    st.progress((current + 1) / len(STEPS))


def show_paragraph(p: dict, evidence: str = "") -> None:
    st.caption(md(p["section"]))
    st.markdown(highlight(p["text"], evidence) if evidence else md(p["text"]))
    st.markdown(f"🔗 [{p['url']}]({p['url']})")


# ---------------------------------------------------------------- 학습 + 출제


def run_learning(question: str, book_id: int | None, index, client) -> None:
    """검색 → 답변 → (인용한 근거가 있으면) 확인 문제 생성까지 한 번에 한다."""
    reset(keep_question=True)
    state = st.session_state
    state.question = question
    with st.spinner("교재에서 근거 문단을 찾는 중…"):
        state.sources = qs.retrieve(question, 5, book_id, index)
    if not state.sources:
        return  # quiz_session.main과 같다: 근거가 없으면 LLM을 부르지 않는다
    with st.spinner("근거 문단으로 답변을 쓰는 중…"):
        state.answer_text = qs.answer(client, question, state.sources)
    cited = qs.cited_sources(state.answer_text, state.sources)
    if not cited:
        return
    with st.spinner("방금 읽은 근거 문단으로 확인 문제를 만드는 중…"):
        state.quiz = qs.make_quiz(
            client,
            question,
            cited,
            index,
            QUIZ_TOTAL,
            QUIZ_FORMATS,
            random.Random(QUIZ_SEED),
        )


def ask_view(index, client, books: dict) -> None:
    labels = {None: "전체 책", **books}
    book_id = st.selectbox(
        "책", list(labels), format_func=lambda b: labels[b], key="book_id"
    )
    asked = None
    for col, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.question_input = example
            asked = example
    # 질문하기 버튼 없이 Enter로 바로 진행한다. 이미 처리한 질문이면 다시 돌리지 않는다
    question = st.text_input(
        "질문", key="question_input", placeholder="질문을 쓰고 Enter"
    ).strip()
    if question and question != st.session_state.get("question"):
        asked = question
    if asked:
        run_learning(asked, book_id, index, client)


def learn_view() -> None:
    state = st.session_state
    if "sources" not in state:
        return
    if not state.sources:
        st.info(
            "교재에서 이 질문의 근거를 찾지 못했어요. 추측해서 답하지 않아요.  \n"
            f"(질문과 가까운 문단이 거리 기준 {qs.MAX_DISTANCE} 안에 없음, LLM 호출 없음)"
        )
        return
    st.markdown("##### 튜터 답변")
    st.markdown(state.answer_text)
    st.markdown("##### 근거 문단")
    for s in state.sources:
        with st.expander(f"[{s.no}] {s.path}  ·  거리 {s.distance:.3f}"):
            show_paragraph({"section": s.path, "text": s.text, "url": s.url})
    if not qs.cited_sources(state.answer_text, state.sources):
        st.info("답변이 인용한 근거가 없어서 확인 문제를 낼 수 없어요.")
    elif not state.get("quiz"):
        st.warning("확인 문제를 만들지 못했어요. 다른 질문으로 다시 해 주세요.")


# ---------------------------------------------------------------- 풀이 + 채점


def judge(client, q: dict, reply: str) -> tuple[bool | None, str]:
    """(정답 여부, 채점 근거). 정답 목록에 있으면 바로, 없으면 LLM이 같은 뜻인지 판정한다."""
    result = qt.grade(q, reply)
    if result is not None:
        return result, "정답 목록과 일치" if result else "답이 비어 있음"
    user = (
        f"[문제]\n{q['question']}\n\n[정답]\n{q['answer']}"
        f" (다른 표기: {', '.join(q['accepted'])})\n\n[학습자 답]\n{reply}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        text = qs.chat(
            client, messages, JUDGE_TEMPERATURE, qs.QUIZ_MAX_TOKENS, JUDGE_SCHEMA
        )
        match = qs.JSON_OBJECT.search(text)
        data = json.loads(match.group()) if match else {}
    except (qs.APIError, json.JSONDecodeError) as e:
        logging.getLogger("demo").warning("AI 채점 실패: %s", e)
        return None, "AI 채점에 실패했어요. 아래 정답과 직접 비교해 주세요."
    if not isinstance(data.get("correct"), bool):
        return None, "AI 채점 결과를 읽지 못했어요. 아래 정답과 직접 비교해 주세요."
    return data["correct"], f"AI 채점: {data.get('reason', '')}"


def quiz_view(client) -> None:
    state = st.session_state
    if not state.get("quiz"):
        return
    q = state.quiz[0]
    st.divider()
    skill = f" · {q['skill']}" if q.get("skill") else ""
    st.caption(f"확인 문제  ·  {qt.FORMAT_LABELS[q['format']]}{skill}")
    st.markdown(f"### {md(q['question'])}")
    if q.get("code", "").strip():
        st.code(q["code"].strip(), language="python")

    if "record" in state:
        return
    # 폼 키는 session_state 키와 겹치면 안 된다 (위젯 키가 되어 값을 대입할 수 없다)
    with st.form("reply_form"):
        reply = st.text_input("답")
        submitted = st.form_submit_button("제출", type="primary")
    if not submitted:
        return
    if not reply.strip():
        st.warning("답을 입력해 주세요.")
        return
    with st.spinner("채점하는 중…"):
        result, reason = judge(client, q, reply.strip())
    state.record = {"reply": reply.strip(), "result": result, "reason": reason}
    st.rerun()


def result_view() -> None:
    state = st.session_state
    if "record" not in state:
        return
    q, rec = state.quiz[0], state.record
    mine = f"내 답: {md(rec['reply'])}"
    if rec["result"] is True:
        st.success(f"정답이에요!  ({mine})")
    elif rec["result"] is False:
        st.error(f"오답이에요.  ({mine})")
    else:
        st.info(f"채점하지 못했어요.  ({mine})")
    st.caption(md(rec["reason"]))

    st.markdown("##### 정답과 해설")
    st.markdown(f"**정답: {md(q['answer'])}**")
    others = [a for a in q.get("accepted", []) if a != q["answer"]]
    if others:
        st.caption("정답으로 인정하는 다른 표기: " + md(", ".join(others)))
    st.markdown(md(q.get("explanation", "")))
    st.markdown("##### 근거 문단")
    show_paragraph(q["paragraph"], q.get("evidence", ""))

    st.divider()
    if st.button("새 질문하기", type="primary"):
        reset()
        st.rerun()


# ---------------------------------------------------------------- 화면


def main() -> None:
    if api_implementation.Type() != "python":
        st.error(
            "chromadb를 불러오려면 protobuf를 순수 Python 구현으로 써야 해요. "
            "프로젝트 루트에서 아래처럼 다시 실행해 주세요."
        )
        st.code(
            '$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"; '
            "python -m streamlit run demo/demo_app.py",
            language="powershell",
        )
        st.stop()
    try:
        qs.get_model()  # 모델 설정이 없으면 임베딩을 불러오기 전에 멈춘다
        index, client, books = load_resources()
    except (
        SystemExit
    ) as e:  # .env의 키·모델 설정이 없을 때 llm_*.py가 SystemExit로 알린다
        st.error(f"LLM 설정을 확인해 주세요: {e}")
        st.stop()

    st.title("📘 학습 확인 퀴즈")
    try:
        ask_view(
            index, client, books
        )  # 질문 버튼을 누르면 이 안에서 답변·출제까지 끝낸다
        stepper()
        learn_view()
        quiz_view(client)
        result_view()
    except qs.APIConnectionError as e:
        st.error(f"{qs.llm.NAME} 연결 실패: {e}")
    except qs.APIStatusError as e:
        st.error(f"{qs.llm.NAME} API 오류 ({e.status_code}): {e.message}")
    except qs.QuizFormatError as e:
        st.error(f"문제 형식 오류: {e}")


main()
