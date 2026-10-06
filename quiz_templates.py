"""퀴즈 유형 템플릿: 출력 스키마, 검증, 화면 표시, 채점, 빈칸 문제 생성.

유형 (--formats 값):
    ox     O/X        LLM 생성
    blank  빈칸 채우기  책 code_example의 pairs에서 코드로 생성 (LLM 미사용)
    mcq    4지선다     LLM 생성
    short  단답형      LLM 생성

LLM 유형은 유형별 배열을 가진 JSON 스키마로 요청하고, 스키마로 표현할 수 없는
규칙(보기 중복, 정답 번호 범위 등)은 validate()에서 다시 검사한다.
"""

import builtins
import random
import re
from collections import Counter

FORMATS = ("ox", "blank", "mcq", "short")
LLM_FORMATS = ("ox", "mcq", "short")
FORMAT_LABELS = {"ox": "O/X", "blank": "빈칸", "mcq": "4지선다", "short": "단답형"}
SKILLS = ("개념 이해", "결과 예측", "함수 선택", "오류 찾기")
MCQ_CHOICES = 4

# 빈칸으로 가릴 수 있는 이름. 책 예제가 직접 만든 이름(FourCal, setdata 등)은
# 외울 가치가 낮아 제외한다. 바로 호출(add(3, 4))은 내장 함수만, 점 호출(s.add(4))은
# 자료형·re 메서드만 허용해야 직접 만든 add 함수가 set.add로 오인되지 않는다
_TRIVIAL = {"print", "input", "help", "exit", "quit"}
_TYPES = (str, list, dict, set, tuple, int, float)
BLANKABLE_CALLS = {
    n for n in dir(builtins) if n.islower() and not n.startswith("_")
} - _TRIVIAL
BLANKABLE_METHODS = (
    {n for t in _TYPES for n in dir(t) if not n.startswith("_")}
    | {n for n in dir(re) if n.islower() and not n.startswith("_")}
    | {n for n in [*dir(re.Match), *dir(re.Pattern)] if not n.startswith("_")}
)
# group(1): 바로 호출하는 이름, group(2): 점 뒤에서 호출하는 메서드 이름
CALL_NAME = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(|\.\s*([A-Za-z_]\w*)\s*\(")
BLANK = "___"
MAX_BLANK_OUTPUT = 60
MAX_CONTEXT_LINES = 3

# 모든 보기가 "A) ", "1. ", "①" 같은 번호로 시작하면 떼어 낸다 (섞으면 번호가 꼬인다)
CHOICE_PREFIX = re.compile(r"^\s*(?:[A-Da-d1-4]\)|[A-Da-d1-4]\.\s|[①②③④])\s*")
OX_TRUE = {"o", "ㅇ", "t", "true", "참", "맞음", "맞다"}
OX_FALSE = {"x", "f", "false", "거짓", "틀림", "틀리다"}


# ---------------------------------------------------------------- 스키마·프롬프트


def _obj(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


_COMMON = {
    "skill": {"type": "string", "enum": list(SKILLS)},
    "concept": {"type": "string"},
    "source": {"type": "integer"},
}
_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}
ITEM_PROPS = {
    "ox": {"statement": _STR, "answer": {"type": "boolean"}, "explanation": _STR},
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

FORMAT_GUIDE = {
    "ox": (
        '"ox": 참/거짓을 판단할 진술문(statement)과 answer(true/false). '
        "거짓 진술은 맞는 문장에서 한 곳만 바꿔 만들어 (예: 왼쪽→오른쪽). "
        "여러 문제면 참과 거짓을 섞어."
    ),
    "mcq": (
        '"mcq": 4지선다. choices는 서로 다른 보기 4개이고 번호(A), 1.)를 붙이지 마. '
        "보기 4개는 모두 같은 형태로 써: 결과 예측이면 모두 실행 결과 값, "
        "함수 선택이면 모두 함수 이름, 개념 이해면 모두 설명 문장. "
        "형태가 다른 보기(예: 값들 사이의 '길이 5의 문자열')는 바로 오답으로 보이니 금지. "
        "오답은 비슷한 함수와의 혼동, 방향·범위 착각 같은 실제 실수로 만들고, "
        "정답만 유난히 길거나 혼자 다른 표현을 쓰지 않게 해. "
        "choice_notes는 choices와 같은 순서로, 보기마다 맞거나 틀린 이유를 한 줄로."
    ),
    "short": (
        '"short": 한 단어나 한 줄 코드로 답하는 단답형. '
        "accepted에는 answer를 포함해 정답으로 인정할 다른 표기를 넣어 "
        '(예: "strip", "strip()").'
    ),
}


def build_schema(counts: dict[str, int]) -> dict:
    # minItems/maxItems는 넣지 않는다: Groq strict 모드에서 검증된 스키마에 없던
    # 키워드라 거절될 수 있다. 문제 수는 프롬프트로 지시하고 코드에서 맞춘다
    return _obj(
        {
            fmt: {"type": "array", "items": _obj({**ITEM_PROPS[fmt], **_COMMON})}
            for fmt in counts
        }
    )


def format_instructions(counts: dict[str, int]) -> str:
    """유형별 규칙과 필드 이름. 스키마만 넘기면 모델이 키 이름을 지어내므로 명시한다."""
    lines = []
    for fmt, n in counts.items():
        fields = ", ".join([*ITEM_PROPS[fmt], *_COMMON])
        lines.append(f"- {FORMAT_GUIDE[fmt]} ({n}문제)\n  필드: {fields}")
    keys = ", ".join(f'"{fmt}"' for fmt in counts)
    lines.append(
        f"출력은 최상위 키가 {keys}인 JSON 객체 하나이고, 각 값은 문제 배열이야. "
        "JSON 앞뒤에 설명이나 검토 문장을 쓰지 마. source는 숫자 하나."
    )
    return "\n".join(lines)


def split_counts(formats: list[str], total: int) -> dict[str, int]:
    """총 문제 수를 유형에 고르게 나눈다. 나머지는 앞 유형부터 하나씩 더한다."""
    base, extra = divmod(total, len(formats))
    counts = {fmt: base + (i < extra) for i, fmt in enumerate(formats)}
    return {fmt: n for fmt, n in counts.items() if n > 0}


# ---------------------------------------------------------------- 검증·정리


def strip_choice_prefixes(choices: list[str]) -> list[str]:
    if all(CHOICE_PREFIX.match(c) for c in choices):
        return [CHOICE_PREFIX.sub("", c, count=1) for c in choices]
    return choices


IDENTIFIER = re.compile(r"[A-Za-z_][\w.]*(\(\))?", re.ASCII)


def _exposed(answer: str, text: str) -> bool:
    """정답(함수 이름 등)이 문제나 코드에 그대로 보이는지.

    "거짓" 같은 일반 단어 정답은 문제 문장에 자연스럽게 나올 수 있어 검사하지 않는다.
    """
    if not IDENTIFIER.fullmatch(answer):
        return False
    core = answer.removesuffix("()").split(".")[-1]  # str.strip() → strip
    return _appears(core, text)


def validate(fmt: str, q: dict) -> str | None:
    """형식 결함 설명. 정상이면 None."""
    if fmt == "ox":
        if not str(q.get("statement", "")).strip():
            return "진술문 없음"
        return (
            None if isinstance(q.get("answer"), bool) else "answer가 true/false가 아님"
        )
    if not str(q.get("question", "")).strip():
        return "문제 없음"
    if fmt == "short":
        answer = str(q.get("answer", "")).strip()
        if not answer:
            return "정답 없음"
        if _exposed(answer, f"{q['question']}\n{q.get('code', '')}"):
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


def normalize(fmt: str, q: dict, source_nos: set[int]) -> dict:
    q["format"] = fmt
    if q.get("skill") not in SKILLS:
        q["skill"] = ""
    if q.get("source") not in source_nos:
        q["source"] = None
    q["concept"] = str(q.get("concept") or "").strip().lower()
    if fmt == "mcq":
        q["choices"] = strip_choice_prefixes([str(c) for c in q["choices"]])
    if fmt == "short":
        accepted = [str(a) for a in q.get("accepted") or []]
        q["accepted"] = [q["answer"], *accepted]
    return q


def shuffle_mcq(q: dict, rng: random.Random) -> None:
    """보기와 보기별 해설을 함께 섞어서 해설이 엉뚱한 보기를 가리키지 않게 한다."""
    paired = list(zip(q["choices"], q["choice_notes"]))
    answer = paired[q["answer_index"]]
    rng.shuffle(paired)
    q["choices"] = [c for c, _ in paired]
    q["choice_notes"] = [n for _, n in paired]
    q["answer_index"] = paired.index(answer)


# ---------------------------------------------------------------- 빈칸 (책 pairs)


def _walk(blocks: list[dict]):
    for block in blocks:
        if block["type"] == "concept_box":
            yield from _walk(block["blocks"])
        else:
            yield block


def _usable_pair(inp: str, out: str) -> bool:
    return (
        "\n" not in inp
        and not inp.rstrip().endswith(":")  # 여러 줄 문장은 파싱이 깨져 있다
        and out.strip() != ""
        and "\n" not in out
        and not out[:1].isspace()
        and len(out) <= MAX_BLANK_OUTPUT
        and "Error" not in out
        and "Traceback" not in out
    )


def _name_pattern(name: str) -> re.Pattern:
    # ASCII 경계: 기본 \w는 한글도 포함해서 "lstrip은"의 lstrip을 놓친다
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.ASCII)


def _appears(name: str, text: str) -> bool:
    return _name_pattern(name).search(text) is not None


DEFINITION = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)", re.ASCII)


def defined_names(chunks) -> set[str]:
    """책 코드가 직접 정의한 함수·클래스 이름. cal.add()처럼 내장 메서드와
    이름이 같은 사용자 정의 메서드를 빈칸에서 빼는 데 쓴다."""
    names: set[str] = set()
    for chunk in chunks:
        for block in _walk(chunk["blocks"]):
            if block["type"] in ("code", "code_example"):
                names.update(DEFINITION.findall(block.get("code", "")))
    return names


def blank_candidates(
    chunk: dict, source_no: int, exclude: set[str] = frozenset()
) -> list[dict]:
    """조각의 대화형 코드 예제에서 함수·메서드 이름을 가린 빈칸 문제를 만든다.

    출력이 없는 앞쪽 입력(대입문 등)은 문맥으로 함께 보여 준다.
    exclude에 든 이름(책이 직접 정의한 이름)은 가리지 않는다.
    """
    items = []
    for block in _walk(chunk["blocks"]):
        if block["type"] != "code_example":
            continue
        context: list[str] = []
        for pair in block.get("pairs") or []:
            inp, out = pair.get("input", ""), pair.get("output", "")
            if not out.strip():
                if "\n" not in inp:
                    context = [*context, inp][-MAX_CONTEXT_LINES:]
                continue
            if not _usable_pair(inp, out):
                continue
            names = [
                m.group(1) or m.group(2)
                for m in CALL_NAME.finditer(inp)
                if (m.group(1) in BLANKABLE_CALLS or m.group(2) in BLANKABLE_METHODS)
                and (m.group(1) or m.group(2)) not in exclude
            ]
            target = next(
                (
                    n
                    for n in reversed(names)  # 바깥 호출보다 마지막(안쪽·메서드) 우선
                    if len(_name_pattern(n).findall(inp)) == 1
                    and not any(_appears(n, line) for line in [*context, out])
                ),
                None,
            )
            if target is None:
                continue
            blanked = _name_pattern(target).sub(BLANK, inp, count=1)
            code = "\n".join([*(f">>> {c}" for c in context), f">>> {blanked}", out])
            items.append(
                {
                    "format": "blank",
                    "skill": "함수 선택",
                    "concept": target.lower(),
                    "question": "빈칸에 들어갈 함수(메서드) 이름은?",
                    "code": code,
                    "answer": target,
                    "accepted": [target],
                    "explanation": f"책 예제에서 {inp} 의 실행 결과는 {out} 이다.",
                    "source": source_no,
                }
            )
    return items


def pick_blanks(
    chunks_by_source: list[tuple[int, dict]],
    n: int,
    rng: random.Random,
    focus: str = "",
    exclude: set[str] = frozenset(),
) -> list[dict]:
    """근거 조각들에서 빈칸 문제를 고른다. 같은 이름은 한 번만 낸다.

    조각에는 학습 주제와 무관한 예제도 섞여 있으므로(strip 조각의 join 등),
    focus(답변 본문)에 나온 이름을 먼저 고른다.
    """
    pool = [
        q
        for no, chunk in chunks_by_source
        for q in blank_candidates(chunk, no, exclude)
    ]
    rng.shuffle(pool)
    pool.sort(key=lambda q: not _appears(q["answer"], focus))  # 안정 정렬
    picked, seen = [], set()
    for q in pool:
        if q["concept"] not in seen:
            picked.append(q)
            seen.add(q["concept"])
        if len(picked) == n:
            break
    return picked


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
    # 단답형은 표현이 다양해 목록에 없어도 맞을 수 있다 → 학습자가 스스로 판단.
    # 빈칸은 책 예제 기준으로 채점한다 (upper/swapcase처럼 결과가 같은 답은 놓친다)
    return None if fmt == "short" and reply else False
