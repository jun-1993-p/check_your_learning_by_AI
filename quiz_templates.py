"""퀴즈 유형 템플릿: 출력 스키마, 검증, 화면 표시, 채점, 빈칸 만들기.

모든 유형을 LLM이 문단 하나에서 문제 하나씩 만든다 (생성은 quiz_session.py):
    ox     O/X
    blank  빈칸 채우기 (LLM이 쓴 새 문장에 ____)
    mcq    4지선다
    short  단답형

스키마로 표현할 수 없는 규칙(보기 중복, 정답 번호 범위, 근거 인용과 정답이 문단에 있는지 등)은
validate()와 check_against()에서 다시 검사한다. 문단 밖 내용으로 만든 문제를 거르는 장치다.
"""

import random
import re
from collections import Counter

FORMATS = ("ox", "blank", "mcq", "short")
FORMAT_LABELS = {"ox": "O/X", "blank": "빈칸", "mcq": "4지선다", "short": "단답형"}
SKILLS = ("개념 이해", "결과 예측", "함수 선택", "오류 찾기")
MCQ_CHOICES = 4

# 문제를 낼 수 없을 때 LLM이 고르는 사유 (eval_blank.py와 같다)
OK_REASON = "없음"
REJECT_REASONS = ("정보 없음", "비유", "다른 언어", "추상적 주장", "개념 불일치")

BLANK = "____"
MAX_ANSWER_CHARS = 30
MIN_EVIDENCE_CHARS = 8  # 근거 인용이 이보다 짧으면 아무 데나 들어맞는다

# 모든 보기가 "A) ", "1. ", "①" 같은 번호로 시작하면 떼어 낸다 (섞으면 번호가 꼬인다)
CHOICE_PREFIX = re.compile(r"^\s*(?:[A-Da-d1-4]\)|[A-Da-d1-4]\.\s|[①②③④])\s*")
OX_TRUE = {"o", "ㅇ", "t", "true", "참", "맞음", "맞다"}
OX_FALSE = {"x", "f", "false", "거짓", "틀림", "틀리다"}


# ---------------------------------------------------------------- 텍스트 비교·빈칸


class BlankError(ValueError):
    """LLM이 쓴 문장·정답으로 빈칸 문제를 만들 수 없음."""


def squash(text: str) -> str:
    """공백·굽은 따옴표·끝 문장부호 차이를 없앤 비교용 문자열."""
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    return re.sub(r"\s+", "", text).rstrip(".!?")


def contains(text: str, snippet: str) -> bool:
    """snippet이 text에 들어 있나 (공백·굽은 따옴표 차이는 무시)."""
    return squash(snippet) in squash(text)


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
    if squash(answer) == squash(sentence):
        raise BlankError("정답이 문장 전체")

    question, count = answer_pattern(answer).subn(BLANK, sentence)
    if count == 0:  # 같은 단어가 없으면 있는 그대로 첫 번째 하나만 가린다
        question, count = sentence.replace(answer, BLANK, 1), 1
    if squash(question.replace(BLANK, answer)) != squash(sentence):
        raise BlankError("채운 문장이 원문과 다름")
    return question, count


def filled(question: str, answer: str) -> str:
    """빈칸을 정답으로 채운 문장."""
    return question.replace(BLANK, answer)


# ---------------------------------------------------------------- 스키마·프롬프트


def _obj(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}
ITEM_PROPS = {
    "ox": {"statement": _STR, "answer": {"type": "boolean"}, "explanation": _STR},
    "blank": {
        "question": _STR,
        "answer": _STR,
        "accepted": _STR_LIST,
        "explanation": _STR,
    },
    "mcq": {
        "question": _STR,
        "code": _STR,
        "choices": _STR_LIST,
        "answer_index": {"type": "integer"},
        "choice_notes": _STR_LIST,
    },
    "short": {
        "question": _STR,
        "code": _STR,
        "answer": _STR,
        "accepted": _STR_LIST,
        "explanation": _STR,
    },
}
# 어떤 유형이든 공통으로 받는 필드. evidence는 문제의 근거를 문단에서 그대로 옮긴 구절이다
COMMON_PROPS = {
    "skill": {"type": "string", "enum": list(SKILLS)},
    "concept": _STR,
    "evidence": _STR,
}
REJECT_PROP = {
    "reject_reason": {"type": "string", "enum": [OK_REASON, *REJECT_REASONS]}
}

FORMAT_GUIDE = {
    "ox": (
        "O/X: 참/거짓을 판단할 진술문(statement)과 answer(true/false). "
        "거짓 진술은 [문단]의 맞는 문장에서 한 곳만 바꿔 만들어 (예: 왼쪽→오른쪽)."
    ),
    "blank": (
        f'빈칸 채우기: question은 [문단]에 근거해 새로 쓴 한 문장이고, 정답 자리는 "{BLANK}"로 써. '
        f'같은 단어가 여러 번 나오면 모두 "{BLANK}"로 가려. '
        "answer는 [문단]에도 나오는 파이썬 개념 용어(조사 제외)이고, question의 다른 곳에는 남아 있으면 안 돼. "
        "accepted에는 answer를 포함해 정답으로 인정할 다른 표기를 넣어."
    ),
    "mcq": (
        "4지선다: choices는 서로 다른 보기 4개이고 번호(A), 1.)를 붙이지 마. "
        "보기 4개는 모두 같은 형태로 써: 결과 예측이면 모두 실행 결과 값, "
        "함수 선택이면 모두 함수 이름, 개념 이해면 모두 설명 문장. "
        "오답은 [문단]을 피상적으로 읽은 학습자가 고를 법하게 만들어: 비슷한 개념과의 혼동, "
        "방향·범위 착각, 문단 속 다른 용어나 조건을 잘못 연결한 실제 실수여야 하고, 누가 봐도 틀린 보기는 안 돼. "
        "보기들의 문법·구체성·어조·정보량을 맞추고, 정답만 유난히 길거나 짧거나 혼자 다른 표현을 쓰지 않게 해. "
        "choice_notes는 choices와 같은 순서로, 보기마다 맞거나 틀린 이유를 한 줄로."
    ),
    "short": (
        "단답형: 한 단어나 한 줄 코드로 답해. answer는 [문단]에 나오는 표현이어야 하고, "
        'accepted에는 answer를 포함해 정답으로 인정할 다른 표기를 넣어 (예: "strip", "strip()").'
    ),
}


def build_schema(fmt: str) -> dict:
    """문제 하나를 받는 스키마. 거부할 수 있도록 reject_reason을 맨 앞에 둔다."""
    return _obj({**REJECT_PROP, **ITEM_PROPS[fmt], **COMMON_PROPS})


def format_instruction(fmt: str) -> str:
    fields = ", ".join(["reject_reason", *ITEM_PROPS[fmt], *COMMON_PROPS])
    return f"- {FORMAT_GUIDE[fmt]}\n  필드: {fields}"


# ---------------------------------------------------------------- 검증·정리


def strip_choice_prefixes(choices: list[str]) -> list[str]:
    if all(CHOICE_PREFIX.match(c) for c in choices):
        return [CHOICE_PREFIX.sub("", c, count=1) for c in choices]
    return choices


IDENTIFIER = re.compile(r"[A-Za-z_][\w.]*(\(\))?", re.ASCII)


def _name_pattern(name: str) -> re.Pattern:
    # ASCII 경계: 기본 \w는 한글도 포함해서 "lstrip은"의 lstrip을 놓친다
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.ASCII)


def _appears(name: str, text: str) -> bool:
    return _name_pattern(name).search(text) is not None


def _exposed(answer: str, text: str) -> bool:
    """정답(함수 이름 등)이 문제나 코드에 그대로 보이는지.

    "거짓" 같은 일반 단어 정답은 문제 문장에 자연스럽게 나올 수 있어 검사하지 않는다.
    """
    if not IDENTIFIER.fullmatch(answer):
        return False
    core = answer.removesuffix("()").split(".")[-1]  # str.strip() → strip
    return _appears(core, text)


def is_rejected(q: dict) -> bool:
    return q.get("reject_reason", OK_REASON) != OK_REASON


def validate(fmt: str, q: dict) -> str | None:
    """형식 결함 설명. 정상이면 None. 거부된 응답은 호출하는 쪽에서 먼저 거른다."""
    if fmt == "ox":
        if not str(q.get("statement", "")).strip():
            return "진술문 없음"
        return (
            None if isinstance(q.get("answer"), bool) else "answer가 true/false가 아님"
        )
    if not str(q.get("question", "")).strip():
        return "문제 없음"
    if fmt in ("short", "blank"):
        answer = str(q.get("answer", "")).strip()
        if not answer:
            return "정답 없음"
        if fmt == "short" and _exposed(answer, f"{q['question']}\n{q.get('code', '')}"):
            return f"정답 노출 ({answer})"
        return None
    choices, notes, idx = q.get("choices"), q.get("choice_notes"), q.get("answer_index")
    if not isinstance(choices, list) or len(choices) != MCQ_CHOICES:
        return f"보기 {len(choices) if isinstance(choices, list) else 0}개"
    # 공백 문제처럼 앞뒤 공백만 다른 보기가 정상이므로 원문 그대로 비교한다
    duplicated = [c for c, n in Counter(map(str, choices)).items() if n > 1]
    if duplicated:
        return f"중복 보기 {duplicated!r}"  # repr로 앞뒤 공백까지 보이게
    if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(choices):
        return f"answer_index 범위 밖 ({idx})"
    if not isinstance(notes, list) or len(notes) != len(choices):
        return "보기별 해설 개수 불일치"
    return None


def check_against(fmt: str, q: dict, text: str) -> str | None:
    """문제가 근거 문단(text)에 실제로 기대고 있는지 검사한다. 정상이면 None.

    - 근거 인용(evidence)이 문단에 그대로 있어야 한다 (문단 밖 지식으로 만든 문제를 거른다).
    - 빈칸·단답형은 정답도 문단에 있어야 한다.
    - 빈칸은 같은 단어를 모두 가려서 q["question"]을 고쳐 쓴다.
    """
    evidence = str(q.get("evidence", "")).strip()
    if len(squash(evidence)) < MIN_EVIDENCE_CHARS:
        return "근거 인용이 없거나 너무 짧음"
    if not contains(text, evidence):
        return "근거 인용이 문단에 없음"
    if fmt == "blank":
        answer = str(q["answer"]).strip()
        try:
            q["question"], q["blank_count"] = make_question(q["question"], answer)
        except BlankError as e:
            return f"빈칸 오류({e})"
        if not contains(text, answer):
            return "정답이 문단에 없음"
    if fmt == "short":
        candidates = [str(q["answer"]), *map(str, q.get("accepted") or [])]
        if not any(contains(text, c) for c in candidates if c.strip()):
            return "정답이 문단에 없음"
    return None


def normalize(fmt: str, q: dict) -> dict:
    q["format"] = fmt
    if q.get("skill") not in SKILLS:
        q["skill"] = ""
    q["concept"] = str(q.get("concept") or "").strip().lower()
    if fmt == "mcq":
        q["choices"] = strip_choice_prefixes([str(c) for c in q["choices"]])
    if fmt in ("short", "blank"):
        accepted = [str(a) for a in q.get("accepted") or []]
        q["accepted"] = list(dict.fromkeys([q["answer"], *accepted]))
    return q


def shuffle_mcq(q: dict, rng: random.Random) -> None:
    """보기와 보기별 해설을 함께 섞어서 해설이 엉뚱한 보기를 가리키지 않게 한다."""
    paired = list(zip(q["choices"], q["choice_notes"]))
    answer = paired[q["answer_index"]]
    rng.shuffle(paired)
    q["choices"] = [c for c, _ in paired]
    q["choice_notes"] = [n for _, n in paired]
    q["answer_index"] = paired.index(answer)


# ---------------------------------------------------------------- 표시·채점


def indent(text: str, prefix: str = "   ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))


def render_question(q: dict) -> str:
    tag = FORMAT_LABELS[q["format"]] + (f" · {q['skill']}" if q.get("skill") else "")
    if q["format"] == "ox":
        return f"[{tag}] {q['statement']}\n   (O 또는 X)"
    lines = [f"[{tag}] {q['question']}"]
    if q.get("code", "").strip():
        lines.append(indent(q["code"].strip()))
    if q["format"] == "mcq":
        lines += [f"   {n}) {c}" for n, c in enumerate(q["choices"], start=1)]
    return "\n".join(lines)


def render_answer(q: dict) -> str:
    fmt = q["format"]
    if fmt == "ox":
        return f"정답: {'O' if q['answer'] else 'X'}\n해설: {q.get('explanation', '')}"
    if fmt == "mcq":
        idx = q["answer_index"]
        notes = [
            f"  {'✔' if n == idx else ' '} {n + 1}) {note}"
            for n, note in enumerate(q["choice_notes"])
        ]
        return f"정답: {idx + 1}) {q['choices'][idx]}\n보기 해설:\n" + "\n".join(notes)
    return f"정답: {q['answer']}\n해설: {q.get('explanation', '')}"


def _norm_text(text: str) -> str:
    text = text.strip().strip("'\"` ").lower()
    text = re.sub(r"\s+", "", text)
    text = text.removeprefix(".")
    return text.removesuffix("()")


def grade(q: dict, reply: str) -> bool | None:
    """자동 채점 결과. 판단할 수 없으면 None (학습자가 스스로 비교)."""
    reply = reply.strip()
    fmt = q["format"]
    if fmt == "ox":
        key = reply.lower()
        if key in OX_TRUE:
            return q["answer"] is True
        if key in OX_FALSE:
            return q["answer"] is False
        return None
    if fmt == "mcq":
        return reply == str(q["answer_index"] + 1)
    if _norm_text(reply) in {_norm_text(a) for a in q["accepted"]}:
        return True
    # 단답형·빈칸은 표현이 다양해 목록에 없어도 맞을 수 있다 → 학습자가 스스로 판단
    return None if reply else False
