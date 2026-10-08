"""수집한 raw HTML을 블록(문단·코드·표 등) 단위로 정제해 chunks.jsonl에 저장한다.

text_ingestion.py가 만든 pages.jsonl을 목록으로 삼고, raw/{id}.html만 읽는다.
네트워크는 사용하지 않는다. 페이지 하나가 조각 하나다 (chunk_id = "{page_id}-01").
입력(pages.jsonl, refine_config.json)이 chunks.jsonl보다 오래됐으면 정제를 건너뛴다.

사용 예:
    python text_refine.py               # .data 아래 모든 책
    python text_refine.py --book-id 1   # 특정 책만
    python text_refine.py --force       # 입력이 그대로여도 다시 정제

책마다 다른 제외 규칙은 프로젝트 루트의 refine_config.json에 둔다 (선택):
    {
      "book1": {
        "exclude_titles": ["되새김 문제"],   # 제목 정규식
        "exclude_ids": [180361],
        "include_ids": [4307]               # 자동 제외 규칙보다 우선
      }
    }
"""

import argparse
import ast
import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from text_ingestion import (
    BASE_URL,
    DATA_ROOT,
    PROJECT_ROOT,
    Paths,
    find_cached_page,
    write_if_changed,
)

# 출제할 내용이 없는 페이지를 본문으로 판별한다 (책 구조와 무관)
MIN_PAGE_CHARS = 300  # 이보다 짧으면 안내·빈 페이지
NAV_PAGE_MAX_CHARS = 800  # 이보다 짧으면서
NAV_LINK_RATIO = 0.1  # 링크 글자 비율이 이 이상이면 하위 목차만 있는 장 제목 페이지

# 어느 책에나 있는 앞뒤 부속 페이지. 번호를 뗀 제목의 앞부분(영문은 전체)과 비교
FRONT_BACK_MATTER = re.compile(
    r"^(?:머리말|서문|들어가며|들어가는 ?글|저자 ?소개|지은이 ?소개|마치며|맺음말"
    r"|나가며|감사의 ?글|참고 ?문헌|책 ?구입|구입 ?안내|(?:주요 ?)?변경 ?이력"
    r"|정답|해답|연습 ?문제)"
    r"|^(?:preface|acknowledgements?|references|bibliography|exercises?|answers?"
    r"|solutions?)$",
    re.IGNORECASE,
)
TITLE_NUMBERING = re.compile(r"^\s*(?:제?\s*\d+\s*장|\d+(?:[-.]\d+)*[.)]?)\s*")
CONFIG_FILE = PROJECT_ROOT / "refine_config.json"

LANG_ALIASES = {"textplain": "plaintext", "no-highlight": "plaintext"}
PYTHON_LANGS = {"python", "py", "python3", "pycon"}
SHELL_LANGS = {"bash", "sh", "shell", "console", "powershell", "bat"}
# 원본(위키독스)이 강조할 줄에 박아 둔 표시
MARK_TAG = re.compile(r"\[\[/?MARK\]\]")
# 명령줄(C:\...> 프롬프트, $ 프롬프트, python 파일.py 실행)
SHELL_LINE = re.compile(
    r"^(?:[A-Za-z]:\\[^>\n]*>|[A-Za-z]:\\\S*\.exe\b|\$ |python3?\s+\S+\.py)", re.MULTILINE
)
# python으로 표시됐지만 실행 결과·표인 블록 (날짜로 시작하는 표, 오류 출력, IDLE 재시작 줄)
OUTPUT_LIKE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}\s|Traceback \(most recent call last\)|=+ RESTART:)", re.MULTILINE
)
# 원본 작성 오류로 단락에 떨어져 나온 코드 울타리 여는 줄 (예: ```{.py}). 이어지는 <pre>가 진짜 코드다
STRAY_FENCE = re.compile(r"`{3}[\w{}.\-]*")
# IPython 세션(`In [3]:` 프롬프트). >>> 대화형과 달리 입출력을 나누지 않고 문법 검사도 하지 않는다
IPYTHON_PROMPT = re.compile(r"^(?:In|Out) ?\[\d+\]:", re.MULTILINE)
# 문법이 안 맞는 블록 중 파이썬 코드다운 줄이 하나도 없으면 코드가 아니라 실행 결과(표·값)다
CODE_TOKEN = re.compile(
    r"^\s*(?:def|class|import|from|for|while|if|elif|else|return|print)\b|\w\s*=[^=]|\w\(.*\)|:\s*$"
)
# 들여쓰기 없는 줄에 나오는 오류 출력만 잡는다 (except 절 코드는 제외)
ERROR_LINE = re.compile(
    r"^([A-Z]\w*(?:Error|Exception|Warning|Interrupt)|StopIteration)\b"
)
SENTENCE_END = re.compile(r"""[.!?다][\"'”’]?$""")
# 출판 원고 형식의 그림·표 설명 (예: "그림 2.9 문자열 인덱싱 예")
CAPTION = re.compile(r"^(?:그림|표|figure|fig\.|table)\s*\d", re.IGNORECASE)

logger = logging.getLogger("text_refine")


# ---------------------------------------------------------------- HTML 파싱
# 수집 모듈(text_ingestion)에서 옮겨 왔다. raw HTML을 읽는 일은 정제 단계가 맡는다

# 본문(div.page-content) 안에 섞여 있는 노이즈
NOISE_SELECTORS = ["div.ad-wrapper", "div.toc", "script", "ins", "style", "legend"]
INLINE_WS = re.compile(r"[ \t\r\n]+")


def extract_body(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.page-content")
    if body is None:
        raise ValueError("본문(div.page-content)을 찾을 수 없음")
    for selector in NOISE_SELECTORS:
        for node in body.select(selector):
            node.decompose()
    return body


def _image_placeholder(img: Tag) -> str:
    alt = (img.get("alt") or "").strip()
    name = alt or Path(img.get("src") or "").name or "image"
    return f"[이미지: {name}]"


def render_inline(node) -> str:
    if isinstance(node, NavigableString):
        return INLINE_WS.sub(" ", str(node))
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return "\n"
    if node.name == "img":
        return _image_placeholder(node)
    inner = "".join(render_inline(c) for c in node.children)
    if node.name == "code":
        return f"`{inner.strip()}`"
    if node.name in ("strong", "b"):
        return f"**{inner.strip()}**" if inner.strip() else ""
    return inner


def _clean_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def render_code(pre: Tag) -> str:
    code = pre.find("code") or pre
    lang = ""
    for cls in code.get("class") or []:
        if cls.startswith("language-"):
            lang = cls.removeprefix("language-")
    body = code.get_text().rstrip("\n")
    return f"```{lang}\n{body}\n```"


def render_list(lst: Tag, indent: int = 0) -> str:
    lines = []
    ordered = lst.name == "ol"
    for i, li in enumerate(lst.find_all("li", recursive=False), start=1):
        marker = f"{i}." if ordered else "-"
        inline = []
        nested = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                nested.append(render_list(child, indent + 1))
            elif isinstance(child, Tag) and child.name == "pre":
                nested.append(render_code(child))
            else:
                inline.append(render_inline(child))
        lines.append(f"{'  ' * indent}{marker} {_clean_lines(''.join(inline))}")
        lines.extend(nested)
    return "\n".join(lines)


def render_table(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [
            _clean_lines(render_inline(c)).replace("\n", " ")
            for c in tr.find_all(["th", "td"])
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------- 블록 변환


def parse_code(pre: Tag) -> dict:
    code_tag = pre.find("code") or pre
    lang = "plaintext"
    for cls in code_tag.get("class") or []:
        if cls.startswith("language-"):
            lang = cls.removeprefix("language-")
    lang = LANG_ALIASES.get(lang, lang)
    raw = code_tag.get_text().rstrip("\n")
    # 원본이 강조하려고 코드 안에 박아 둔 [[MARK]] 표시는 코드가 아니라서 뗀다 (뗀 사실은 marked에 남긴다)
    code = MARK_TAG.sub("", raw)

    block: dict = {"type": "code", "lang": lang, "code": code}
    if code != raw:
        block["marked"] = True
    # 오류 출력과 >>> 대화형 예제는 파이썬 코드에서만 해석한다.
    # 실행 결과는 보통 plaintext로 표시되므로 plaintext도 포함한다
    if lang in PYTHON_LANGS or lang == "plaintext":
        errors = [m.group(1) for line in code.split("\n") if (m := ERROR_LINE.match(line))]
        if errors:
            block["error"] = errors[-1]
        if code.lstrip().startswith(">>>"):
            block["type"] = "code_example"
            block["pairs"] = split_interactive(code)
    block["kind"], block["syntax_ok"], block["issues"] = classify_code(block)
    return block


def split_interactive(code: str) -> list[dict]:
    """>>> 대화형 코드를 입력·출력 쌍으로 나눈다.

    책에는 `...` 없이 들여쓰기만 한 이어쓰기 줄이 있다. 입력이 `:`로 끝난 뒤의 들여쓴 줄(사이의
    빈 줄 포함)은 출력이 아니라 입력의 이어쓰기로 본다.
    """
    pairs: list[dict] = []
    lines = code.split("\n")
    in_block = False  # 앞선 입력이 복합문(`:`로 끝남)이라 들여쓴 줄이 이어지는 중
    for n, line in enumerate(lines):
        if line.startswith(">>>"):
            pairs.append({"input": [line[4:]], "output": []})
            in_block = line[4:].rstrip().endswith(":")
        elif pairs and line.startswith("...") and not pairs[-1]["output"]:
            pairs[-1]["input"].append(line[4:])
            in_block = line[4:].rstrip().endswith(":") or in_block
        elif pairs and in_block and not pairs[-1]["output"] and continues_block(lines, n):
            pairs[-1]["input"].append(line)
        elif pairs:
            pairs[-1]["output"].append(line)
            in_block = False
    result = [
        {"input": "\n".join(p["input"]), "output": "\n".join(p["output"]).rstrip()}
        for p in pairs
    ]
    return [p for p in result if p["input"].strip() or p["output"]]  # 빈 >>> 줄 제외


def continues_block(lines: list[str], n: int) -> bool:
    """lines[n]이 복합문 입력의 이어쓰기 줄인가: 들여쓴 줄이거나, 뒤에 들여쓴 줄이 또 이어지는 빈 줄."""
    line = lines[n]
    if line[:1] in (" ", "\t"):
        return True
    if line.strip():
        return False
    for nxt in lines[n + 1 :]:
        if nxt.strip():
            return nxt[:1] in (" ", "\t")
    return False


def classify_code(block: dict) -> tuple[str, bool | None, list[str]]:
    """코드 블록의 종류, 문법 검사 결과, 문제점 목록.

    종류: repl(>>> 대화형) / ipython(In [n]: 세션) / python(문법이 맞는 코드) / shell(명령줄) / output(실행 결과·표) /
    pseudo(python으로 표시됐지만 문법이 안 맞는 의사코드·문법 설명·불완전한 코드) / other(다른 언어).
    syntax_ok는 파이썬 코드로 읽을 수 있는 블록만 True/False이고 나머지는 None이다.
    """
    lang, code = block["lang"], block["code"]
    if lang in SHELL_LANGS or SHELL_LINE.search(code):
        return "shell", None, []
    if lang not in PYTHON_LANGS and lang != "plaintext":
        return "other", None, []
    if IPYTHON_PROMPT.search(code):
        return "ipython", None, []
    if block["type"] == "code_example":
        source = "\n".join(p["input"] for p in block.get("pairs", []))
        kind = "repl"
    elif lang == "plaintext" or OUTPUT_LIKE.search(code):
        return "output", None, []
    else:
        source, kind = code, "python"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "\d" 같은 이스케이프 경고는 문법 오류가 아니다
            ast.parse(source)
    except (SyntaxError, ValueError) as e:
        msg = getattr(e, "msg", str(e))
        issue = "truncated" if "expected an indented block" in msg else "syntax_error"
        if kind == "python" and not any(CODE_TOKEN.search(l) for l in code.split("\n")):
            return "output", None, []
        return ("repl" if kind == "repl" else "pseudo"), False, [issue]
    return kind, True, []


def parse_table(table: Tag) -> dict:
    rows = [
        [
            _clean_lines(render_inline(c)).replace("\n", " ")
            for c in tr.find_all(["th", "td"])
        ]
        for tr in table.find_all("tr")
    ]
    has_header = table.find("th") is not None
    return {
        "type": "table",
        "header": rows[0] if has_header and rows else [],
        "rows": rows[1:] if has_header else rows,
        "_text": render_table(table),
    }


def _bold_only(p: Tag) -> bool:
    strong = p.find(["strong", "b"])
    return strong is not None and p.get_text(strip=True) == strong.get_text(strip=True)


def parse_blocks(container: Tag) -> list[dict]:
    blocks: list[dict] = []
    for node in container.children:
        if isinstance(node, NavigableString):
            text = _clean_lines(render_inline(node))
            if text:
                blocks.append({"type": "paragraph", "text": text})
            continue
        if not isinstance(node, Tag):
            continue
        name = node.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _clean_lines(render_inline(node))
            if text:
                blocks.append(
                    {
                        "type": "heading",
                        "level": int(name[1]),
                        "text": text,
                        "anchor": node.get("id"),
                    }
                )
        elif name == "p":
            text = _clean_lines(render_inline(node))
            if not text or STRAY_FENCE.fullmatch(text):
                continue
            plain = text.strip("*")
            # "표 2.1은 ~이다." 같은 본문 문장은 캡션으로 보지 않는다
            is_caption = CAPTION.match(plain) and (
                _bold_only(node) or not SENTENCE_END.search(plain)
            )
            if is_caption:
                blocks.append({"type": "caption", "text": plain})
            elif _bold_only(node):
                kind = "key_point" if SENTENCE_END.search(plain) else "subheading"
                blocks.append({"type": kind, "text": plain})
            else:
                blocks.append({"type": "paragraph", "text": text})
        elif name == "pre":
            blocks.append(parse_code(node))
        elif name in ("ul", "ol"):
            blocks.append({"type": "list", "text": render_list(node)})
        elif name == "table":
            blocks.append(parse_table(node))
        elif name == "blockquote":
            inner = "\n\n".join(render_block(b) for b in parse_blocks(node))
            if inner:
                blocks.append({"type": "note", "text": inner})
        elif name == "fieldset":
            inner = parse_blocks(node)
            title = ""
            if inner and inner[0]["type"] in ("subheading", "key_point"):
                title = inner.pop(0)["text"]
            if inner:
                blocks.append({"type": "concept_box", "title": title, "blocks": inner})
        elif name in ("div", "section"):
            blocks.extend(parse_blocks(node))
        elif name != "hr":
            text = _clean_lines(render_inline(node))
            if text:
                blocks.append({"type": "paragraph", "text": text})
    return blocks


def render_block(block: dict) -> str:
    """LLM 프롬프트에 넣기 좋은 텍스트로 블록을 표현한다."""
    kind = block["type"]
    if kind == "heading":
        return f"{'#' * block['level']} {block['text']}"
    if kind in ("key_point", "subheading"):
        return f"**{block['text']}**"
    if kind in ("code", "code_example"):
        return f"```{block['lang']}\n{block['code']}\n```"
    if kind == "table":
        return block["_text"]
    if kind == "note":
        return "\n".join(f"> {line}".rstrip() for line in block["text"].split("\n"))
    if kind == "concept_box":
        inner = "\n\n".join(render_block(b) for b in block["blocks"])
        title = f"[개념] {block['title']}" if block["title"] else "[개념]"
        return f"{title}\n{inner}"
    return block["text"]


# ---------------------------------------------------------------- 실행


def load_rules(source_dir: str) -> dict:
    """refine_config.json에서 이 책(book1, unbound 등)의 예외 규칙을 읽는다."""
    rules: dict = {"exclude_titles": [], "exclude_ids": [], "include_ids": []}
    if not CONFIG_FILE.exists():
        return rules
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        rules.update(config.get(source_dir, {}))
        rules["exclude_titles"] = [re.compile(p) for p in rules["exclude_titles"]]
    except (json.JSONDecodeError, re.error) as e:
        raise ValueError(f"{CONFIG_FILE} 형식 오류: {e}") from e
    return rules


def page_stats(body: Tag) -> tuple[int, float]:
    """본문 글자 수와 그중 링크 글자의 비율."""
    text_len = len(" ".join(body.get_text(" ").split()))
    link_len = sum(len(" ".join(a.get_text(" ").split())) for a in body.find_all("a"))
    return text_len, link_len / max(text_len, 1)


def exclusion_reason(page: dict, body: Tag, rules: dict) -> str | None:
    if page["id"] in rules["include_ids"]:
        return None
    if page["id"] in rules["exclude_ids"]:
        return "설정 exclude_ids"
    title = page.get("title", "")
    if FRONT_BACK_MATTER.search(TITLE_NUMBERING.sub("", title)):
        return "부속 페이지 제목"
    for pattern in rules["exclude_titles"]:
        if pattern.search(title):
            return f"설정 exclude_titles: {pattern.pattern}"
    chars, link_ratio = page_stats(body)
    if chars < MIN_PAGE_CHARS:
        return f"본문 {chars}자"
    if chars < NAV_PAGE_MAX_CHARS and link_ratio >= NAV_LINK_RATIO:
        return f"목차형 페이지 (본문 {chars}자, 링크 {link_ratio:.0%})"
    return None


def _strip_private(block: dict) -> dict:
    clean = {k: v for k, v in block.items() if not k.startswith("_")}
    if "blocks" in clean:
        clean["blocks"] = [_strip_private(b) for b in clean["blocks"]]
    return clean


def assign_code_ids(blocks: list[dict], chunk_id: str, prefix: str = "") -> None:
    """코드 블록마다 code_id를 붙인다. 문단 id(`{chunk_id}#b{블록 번호}`)와 같은 번호 체계이고,
    개념 상자 안의 블록은 `#b{상자 번호}.{안쪽 번호}`다."""
    for i, block in enumerate(blocks):
        name = f"{prefix}{i}"
        if block["type"] in ("code", "code_example"):
            block["code_id"] = f"{chunk_id}#b{name}"
        elif block["type"] == "concept_box":
            assign_code_ids(block["blocks"], chunk_id, f"{name}.")


def code_rows(chunk: dict) -> list[dict]:
    """코드 저장소(code_blocks.jsonl)에 쓸 줄: 조각의 코드 블록마다 하나."""
    rows: list[dict] = []

    def walk(blocks: list[dict]) -> None:
        for block in blocks:
            if block["type"] == "concept_box":
                walk(block["blocks"])
            elif block["type"] in ("code", "code_example"):
                rows.append(
                    {
                        "code_id": block["code_id"],
                        "chunk_id": chunk["chunk_id"],
                        "page_id": chunk["page_id"],
                        "book_id": chunk["book_id"],
                        **{k: v for k, v in block.items() if k != "code_id"},
                    }
                )

    walk(chunk["blocks"])
    return rows


def make_chunk(page: dict, body: Tag) -> dict | None:
    """페이지 하나를 조각 하나로 만든다. 본문 블록이 없으면 None.

    문단(A+) 단위가 소제목 경계를 스스로 처리하므로 조각을 더 나누지 않는다.
    path는 페이지 경로(목차 breadcrumb)만 담는다: 페이지 관문이 페이지 단위로 동작한다.
    """
    blocks = parse_blocks(body)
    if not blocks:
        return None
    assign_code_ids(blocks, f"{page['id']}-01")
    return {
        "chunk_id": f"{page['id']}-01",
        "page_id": page["id"],
        "book_id": page.get("book_id"),
        "book_title": page.get("book_title"),
        "page_title": page.get("title"),
        "path": list(page.get("breadcrumb", [])),
        "source_url": f"{BASE_URL}/{page['id']}",
        "blocks": [_strip_private(b) for b in blocks],
    }


def load_pages(pages_file: Path) -> list[dict]:
    pages = []
    with pages_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("%s %d번째 줄 무시: %s", pages_file, line_no, e)
    return sorted(pages, key=lambda p: (p.get("order", -1), p["id"]))


def is_up_to_date(book_dir: Path) -> bool:
    """chunks.jsonl이 입력(pages.jsonl, refine_config.json)보다 새로우면 다시 정제할 필요가 없다.

    raw HTML만 바뀐 경우는 알 수 없으니 그때는 force(--force, 파이프라인은 --rebuild)로 돌린다.
    """
    chunks_file = book_dir / "chunks.jsonl"
    if not chunks_file.exists():
        return False
    built = chunks_file.stat().st_mtime
    inputs = [book_dir / "pages.jsonl", CONFIG_FILE]
    return all(p.stat().st_mtime <= built for p in inputs if p.exists())


def refine_book(book_dir: Path, force: bool = False) -> None:
    pages_file = book_dir / "pages.jsonl"
    chunks_file = book_dir / "chunks.jsonl"
    excluded_file = book_dir / "excluded.json"
    raw_dir = book_dir / "raw"
    if not force and is_up_to_date(book_dir):
        logger.info(
            "%s: 입력이 그대로라 정제를 건너뜀 (--force로 다시 정제)", book_dir.name
        )
        return
    rules = load_rules(book_dir.name)

    all_chunks: list[dict] = []
    excluded: list[dict] = []
    for page in load_pages(pages_file):
        raw_file = raw_dir / f"{page['id']}.html"
        if not raw_file.exists():
            raw_file = find_cached_page(page["id"])
        if raw_file is None:
            logger.warning("raw HTML 없음: %s", page["id"])
            continue
        try:
            body = extract_body(raw_file.read_text(encoding="utf-8"))
            if reason := exclusion_reason(page, body, rules):
                excluded.append(
                    {"id": page["id"], "title": page.get("title"), "reason": reason}
                )
                continue
            if chunk := make_chunk(page, body):
                all_chunks.append(chunk)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.warning("정제 실패 %s: %s", page["id"], e)

    # 잘못 제외된 페이지를 찾을 수 있도록 사유를 남긴다
    write_if_changed(
        excluded_file, json.dumps(excluded, ensure_ascii=False, indent=2) + "\n"
    )
    # 코드는 조각 안의 블록과 별도로 저장소에도 둔다 (종류·문법 검사 결과로 골라 쓰려고)
    rows = [row for c in all_chunks for row in code_rows(c)]
    write_if_changed(
        book_dir / "code_blocks.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    kinds = Counter(r["kind"] for r in rows)
    logger.info("%s: 코드 블록 %d개 %s", book_dir.name, len(rows), dict(kinds))
    content = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks)
    changed = write_if_changed(chunks_file, content)
    if not changed:
        chunks_file.touch()  # 다음 실행에서 "입력보다 새로움"으로 판단하도록 시각을 갱신한다
    logger.info(
        "%s: 조각 %d개, 제외 페이지 %d개 (%s)%s",
        book_dir.name,
        len(all_chunks),
        len(excluded),
        excluded_file.name,
        "" if changed else " - 변경 없음",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="raw HTML을 퀴즈용 개념 조각으로 정제")
    parser.add_argument("--book-id", type=int, default=None, help="생략하면 모든 책")
    parser.add_argument(
        "--force", action="store_true", help="입력이 그대로여도 다시 정제"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.book_id is not None:
        book_dirs = [Paths.for_book(args.book_id).book_dir]
    else:
        book_dirs = sorted(p.parent for p in DATA_ROOT.glob("*/pages.jsonl"))
    for book_dir in book_dirs:
        if not (book_dir / "pages.jsonl").exists():
            logger.error("pages.jsonl 없음: %s", book_dir)
            continue
        refine_book(book_dir, args.force)


if __name__ == "__main__":
    main()
