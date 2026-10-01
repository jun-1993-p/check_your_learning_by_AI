"""수집한 raw HTML을 퀴즈 출제용 개념 조각(chunks.jsonl)으로 정제한다.

text_ingestion.py가 만든 pages.jsonl을 목록으로 삼고, raw/{id}.html만 읽는다.
네트워크는 사용하지 않는다.

사용 예:
    python text_refine.py               # .data 아래 모든 책
    python text_refine.py --book-id 1   # 특정 책만

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
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import NavigableString, Tag

from text_ingestion import (
    BASE_URL,
    DATA_ROOT,
    PROJECT_ROOT,
    Paths,
    _clean_lines,
    extract_body,
    find_cached_page,
    move_to_archive,
    normalize,
    render_inline,
    render_list,
    render_table,
)

DEFAULT_MIN_CHARS = 300  # 이보다 짧은 조각은 앞 조각에 합친다
DEFAULT_MAX_CHARS = 4000  # 이보다 긴 h2 조각은 h3 기준으로 다시 나눈다

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
# 들여쓰기 없는 줄에 나오는 오류 출력만 잡는다 (except 절 코드는 제외)
ERROR_LINE = re.compile(
    r"^([A-Z]\w*(?:Error|Exception|Warning|Interrupt)|StopIteration)\b"
)
SENTENCE_END = re.compile(r"""[.!?다][\"'”’]?$""")
# 출판 원고 형식의 그림·표 설명 (예: "그림 2.9 문자열 인덱싱 예")
CAPTION = re.compile(r"^(?:그림|표|figure|fig\.|table)\s*\d", re.IGNORECASE)

logger = logging.getLogger("text_refine")


@dataclass
class Section:
    headings: list[dict] = field(default_factory=list)  # {"level", "text", "anchor"}
    blocks: list[dict] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(render_block(b) for b in self.blocks)


# ---------------------------------------------------------------- 블록 변환


def parse_code(pre: Tag) -> dict:
    code_tag = pre.find("code") or pre
    lang = "plaintext"
    for cls in code_tag.get("class") or []:
        if cls.startswith("language-"):
            lang = cls.removeprefix("language-")
    lang = LANG_ALIASES.get(lang, lang)
    code = code_tag.get_text().rstrip("\n")

    block: dict = {"type": "code", "lang": lang, "code": code}
    # 오류 출력과 >>> 대화형 예제는 파이썬 코드에서만 해석한다.
    # 실행 결과는 보통 plaintext로 표시되므로 plaintext도 포함한다
    if lang not in PYTHON_LANGS and lang != "plaintext":
        return block
    errors = [m.group(1) for line in code.split("\n") if (m := ERROR_LINE.match(line))]
    if errors:
        block["error"] = errors[-1]
    if code.lstrip().startswith(">>>"):
        block["type"] = "code_example"
        block["pairs"] = split_interactive(code)
    return block


def split_interactive(code: str) -> list[dict]:
    """>>> 대화형 코드를 입력·출력 쌍으로 나눈다."""
    pairs: list[dict] = []
    for line in code.split("\n"):
        if line.startswith(">>>"):
            pairs.append({"input": [line[4:]], "output": []})
        elif pairs and line.startswith("...") and not pairs[-1]["output"]:
            pairs[-1]["input"].append(line[4:])
        elif pairs:
            pairs[-1]["output"].append(line)
    result = [
        {"input": "\n".join(p["input"]), "output": "\n".join(p["output"]).rstrip()}
        for p in pairs
    ]
    return [p for p in result if p["input"].strip() or p["output"]]  # 빈 >>> 줄 제외


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
            if not text:
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


# ---------------------------------------------------------------- 분할


def split_at(blocks: list[dict], level: int, base: list[dict]) -> list[Section]:
    """지정한 레벨의 제목에서 블록 목록을 나눈다."""
    sections = [Section(headings=list(base))]
    for block in blocks:
        if block["type"] == "heading" and block["level"] == level:
            sections.append(Section(headings=[*base, block]))
        sections[-1].blocks.append(block)
    return [s for s in sections if s.blocks]


def chunk_page(blocks: list[dict], min_chars: int, max_chars: int) -> list[Section]:
    sections: list[Section] = []
    for section in split_at(blocks, 2, []):
        if len(section.text()) > max_chars:
            group = split_at(section.blocks, 3, section.headings)
            sections.extend(merge_short(group, min_chars))
        else:
            sections.append(section)
    # h2 하나가 통째로 짧은 경우만 h2 경계를 넘어 합친다
    return merge_short(sections, min_chars)


def _join(into: Section, other: Section, prepend: bool = False) -> None:
    if prepend:
        into.blocks[:0] = other.blocks
        rest = [h for h in into.headings if h not in other.headings]
        into.headings[:] = [*other.headings, *rest]
    else:
        into.blocks.extend(other.blocks)
        into.headings.extend(h for h in other.headings if h not in into.headings)


def merge_short(sections: list[Section], min_chars: int) -> list[Section]:
    """짧은 조각을 합친다. 맨 앞(도입부)은 뒤 조각에, 나머지는 앞 조각에 붙인다."""
    merged: list[Section] = []
    carry: Section | None = None
    for i, section in enumerate(sections):
        if carry:
            _join(section, carry, prepend=True)
            carry = None
        short = len(section.text()) < min_chars
        if short and merged:
            _join(merged[-1], section)
        elif short and i < len(sections) - 1:
            carry = section
        else:
            merged.append(section)
    return merged


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


def write_if_changed(path: Path, content: str) -> bool:
    """내용이 바뀐 경우에만 기존 파일을 archive로 옮기고 새로 쓴다."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    move_to_archive(path)
    path.write_text(content, encoding="utf-8")
    return True


def _strip_private(block: dict) -> dict:
    clean = {k: v for k, v in block.items() if not k.startswith("_")}
    if "blocks" in clean:
        clean["blocks"] = [_strip_private(b) for b in clean["blocks"]]
    return clean


def make_chunks(page: dict, body: Tag, min_chars: int, max_chars: int) -> list[dict]:
    blocks = parse_blocks(body)
    chunks = []
    for n, section in enumerate(chunk_page(blocks, min_chars, max_chars), start=1):
        text = normalize(section.text())
        top = [h for h in section.headings if h["level"] == 2][:1]
        sub = [h for h in section.headings if h["level"] == 3][:1]
        # 가장 구체적인 제목(h3 > h2)의 앵커로 연결한다
        anchor = next((h["anchor"] for h in [*sub, *top] if h["anchor"]), None)
        url = f"{BASE_URL}/{page['id']}" + (f"#{anchor}" if anchor else "")
        chunks.append(
            {
                "chunk_id": f"{page['id']}-{n:02d}",
                "page_id": page["id"],
                "book_id": page.get("book_id"),
                "book_title": page.get("book_title"),
                "page_title": page.get("title"),
                "path": [*page.get("breadcrumb", []), *(h["text"] for h in top + sub)],
                "headings": [h["text"] for h in section.headings],
                "source_url": url,
                "image_dependent": "[이미지:" in text,
                "has_error_example": any("error" in b for b in section.blocks),
                "blocks": [_strip_private(b) for b in section.blocks],
                "text": text,
                "char_count": len(text),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return chunks


def load_pages(pages_file: Path) -> list[dict]:
    pages = []
    with pages_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("%s %d번째 줄 무시: %s", pages_file, line_no, e)
    return sorted(pages, key=lambda p: (p.get("order", -1), p["id"]))


def refine_book(book_dir: Path, min_chars: int, max_chars: int) -> None:
    pages_file = book_dir / "pages.jsonl"
    chunks_file = book_dir / "chunks.jsonl"
    excluded_file = book_dir / "excluded.json"
    raw_dir = book_dir / "raw"
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
            all_chunks.extend(make_chunks(page, body, min_chars, max_chars))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.warning("정제 실패 %s: %s", page["id"], e)

    # 잘못 제외된 페이지를 찾을 수 있도록 사유를 남긴다
    write_if_changed(
        excluded_file, json.dumps(excluded, ensure_ascii=False, indent=2) + "\n"
    )
    content = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks)
    changed = write_if_changed(chunks_file, content)
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
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
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
        refine_book(book_dir, args.min_chars, args.max_chars)


if __name__ == "__main__":
    main()
