"""수집한 raw HTML을 퀴즈 출제용 개념 조각(chunks.jsonl)으로 정제한다.

text_ingestion.py가 만든 pages.jsonl을 목록으로 삼고, raw/{id}.html만 읽는다.
네트워크는 사용하지 않는다.

사용 예:
    python text_refine.py               # .data 아래 모든 책
    python text_refine.py --book-id 1   # 특정 책만
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

# 출제 대상에서 제외할 페이지 (장 제목 페이지는 depth 0으로 따로 제외)
EXCLUDE_TITLE = re.compile(
    r"되새김 문제|정답 및 풀이|면허 시험|머리말|저자소개|동영상 강의|소스 코드"
    r"|구입 안내|변경이력|^마치며"
)
LANG_ALIASES = {"textplain": "plaintext", "no-highlight": "plaintext"}
# 들여쓰기 없는 줄에 나오는 오류 출력만 잡는다 (except 절 코드는 제외)
ERROR_LINE = re.compile(
    r"^([A-Z]\w*(?:Error|Exception|Warning|Interrupt)|StopIteration)\b"
)
SENTENCE_END = re.compile(r"""[.!?다][\"'”’]?$""")

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
    return [
        {"input": "\n".join(p["input"]), "output": "\n".join(p["output"]).rstrip()}
        for p in pairs
    ]


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
            if _bold_only(node):
                plain = text.strip("*")
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


def is_excluded(page: dict) -> str | None:
    if page.get("depth") == 0:
        return "장 제목 페이지"
    if EXCLUDE_TITLE.search(page.get("title", "")):
        return "출제 제외 페이지"
    return None


def _strip_private(block: dict) -> dict:
    clean = {k: v for k, v in block.items() if not k.startswith("_")}
    if "blocks" in clean:
        clean["blocks"] = [_strip_private(b) for b in clean["blocks"]]
    return clean


def make_chunks(page: dict, html: str, min_chars: int, max_chars: int) -> list[dict]:
    blocks = parse_blocks(extract_body(html))
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
    raw_dir = book_dir / "raw"

    all_chunks: list[dict] = []
    excluded = 0
    for page in load_pages(pages_file):
        if reason := is_excluded(page):
            logger.info("제외 %s %s (%s)", page["id"], page.get("title"), reason)
            excluded += 1
            continue
        raw_file = raw_dir / f"{page['id']}.html"
        if not raw_file.exists():
            raw_file = find_cached_page(page["id"])
        if raw_file is None:
            logger.warning("raw HTML 없음: %s", page["id"])
            continue
        try:
            html = raw_file.read_text(encoding="utf-8")
            all_chunks.extend(make_chunks(page, html, min_chars, max_chars))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.warning("정제 실패 %s: %s", page["id"], e)

    content = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks)
    if chunks_file.exists() and chunks_file.read_text(encoding="utf-8") == content:
        logger.info("%s: 변경 없음 (조각 %d개)", book_dir.name, len(all_chunks))
        return
    move_to_archive(chunks_file)
    chunks_file.write_text(content, encoding="utf-8")
    logger.info(
        "%s: 조각 %d개 (제외 페이지 %d개) -> %s",
        book_dir.name,
        len(all_chunks),
        excluded,
        chunks_file,
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
