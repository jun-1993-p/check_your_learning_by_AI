"""위키독스 책의 목차와 본문을 수집해 정제된 텍스트(JSONL)로 저장한다.

사용 예:
    python text_ingestion.py                 # 수집 (캐시가 있으면 재사용)
    python text_ingestion.py --offline       # 네트워크 없이 캐시만으로 재정제
    python text_ingestion.py --refresh       # 기존 결과를 archive로 옮기고 재수집
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

BASE_URL = "https://wikidocs.net"
DATA_ROOT = Path(".data")
ARCHIVE_DIR = Path("archive")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_DELAY_SEC = 10.0
MAX_RETRIES = 3
TIMEOUT_SEC = 15

# 본문(div.page-content) 안에 섞여 있는 노이즈
NOISE_SELECTORS = ["div.ad-wrapper", "div.toc", "script", "ins", "style", "legend"]
ZERO_WIDTH = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
INLINE_WS = re.compile(r"[ \t\r\n]+")

logger = logging.getLogger("text_ingestion")


class FetchBlockedError(Exception):
    """403/429 응답. 우회하지 않고 실행을 중단한다."""


@dataclass
class TocItem:
    id: int
    parent_id: int | None
    depth: int
    order: int
    title: str


@dataclass
class PageRecord:
    id: int
    parent_id: int | None
    depth: int
    order: int
    title: str
    breadcrumb: list[str]
    url: str
    text: str
    fetched_at: str
    content_hash: str


@dataclass
class Paths:
    book_dir: Path
    raw_dir: Path
    pages_file: Path
    failed_file: Path

    @classmethod
    def for_book(cls, book_id: int) -> "Paths":
        book_dir = DATA_ROOT / f"book{book_id}"
        return cls(
            book_dir=book_dir,
            raw_dir=book_dir / "raw",
            pages_file=book_dir / "pages.jsonl",
            failed_file=book_dir / "failed.json",
        )


# ---------------------------------------------------------------- 수집


def build_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Referer": f"{BASE_URL}/",
    }


def fetch_html(url: str, headers: dict[str, str]) -> str:
    """URL을 GET 한다. 일시적 오류는 지수 백오프로 재시도한다."""
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                raise FetchBlockedError(f"{url} -> HTTP {e.code}") from e
            if e.code < 500 or attempt == MAX_RETRIES:
                raise
            logger.warning("HTTP %s (%d/%d): %s", e.code, attempt, MAX_RETRIES, url)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning("%s (%d/%d): %s", e, attempt, MAX_RETRIES, url)
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def load_or_fetch(
    url: str, cache_file: Path, headers: dict[str, str], offline: bool
) -> tuple[str, bool]:
    """캐시가 있으면 읽고, 없으면 받아서 캐시에 저장한다. (html, 네트워크 사용 여부)"""
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8"), False
    if offline:
        raise FileNotFoundError(f"캐시 없음 (offline): {cache_file}")
    html = fetch_html(url, headers)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(html, encoding="utf-8")
    return html, True


# ---------------------------------------------------------------- 파싱


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def parse_toc(html: str) -> list[TocItem]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for order, a in enumerate(soup.select("a.toc-item")):
        page_id = _to_int(a.get("data-id"))
        if page_id is None:
            continue
        items.append(
            TocItem(
                id=page_id,
                parent_id=_to_int(a.get("data-parent-id")),
                depth=_to_int(a.get("data-depth")) or 0,
                order=order,
                title=normalize(a.get_text(" ", strip=True)),
            )
        )
    return items


def build_breadcrumb(item: TocItem, by_id: dict[int, TocItem]) -> list[str]:
    path = [item.title]
    seen = {item.id}
    parent = by_id.get(item.parent_id) if item.parent_id else None
    while parent and parent.id not in seen:
        path.append(parent.title)
        seen.add(parent.id)
        parent = by_id.get(parent.parent_id) if parent.parent_id else None
    return path[::-1]


def extract_body(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.page-content")
    if body is None:
        raise ValueError("본문(div.page-content)을 찾을 수 없음")
    for selector in NOISE_SELECTORS:
        for node in body.select(selector):
            node.decompose()
    return body


# ---------------------------------------------------------------- 정제


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    return ZERO_WIDTH.sub("", text)


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


def render_blocks(container: Tag) -> list[str]:
    blocks: list[str] = []
    inline_buf: list[str] = []

    def flush() -> None:
        text = _clean_lines("".join(inline_buf))
        if text:
            blocks.append(text)
        inline_buf.clear()

    for node in container.children:
        if not isinstance(node, Tag):
            inline_buf.append(render_inline(node))
            continue
        name = node.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            flush()
            title = _clean_lines(render_inline(node))
            if title:
                blocks.append(f"{'#' * int(name[1])} {title}")
        elif name == "p":
            flush()
            text = _clean_lines(render_inline(node))
            if text:
                blocks.append(text)
        elif name == "pre":
            flush()
            blocks.append(render_code(node))
        elif name in ("ul", "ol"):
            flush()
            blocks.append(render_list(node))
        elif name == "table":
            flush()
            blocks.append(render_table(node))
        elif name == "blockquote":
            flush()
            inner = "\n\n".join(render_blocks(node))
            blocks.append("\n".join(f"> {line}".rstrip() for line in inner.split("\n")))
        elif name in ("div", "fieldset", "section"):
            flush()
            blocks.extend(render_blocks(node))
        elif name == "hr":
            flush()
        else:
            inline_buf.append(render_inline(node))
    flush()
    return blocks


def clean_text(body: Tag) -> str:
    return normalize("\n\n".join(render_blocks(body)))


# ---------------------------------------------------------------- 저장


def archive_path(path: Path) -> Path:
    """archive/ 안에 날짜·시각을 붙인 이동 대상 경로를 만든다."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    name = f"{path.parent.name}_{path.stem}_{stamp}{path.suffix}"
    return ARCHIVE_DIR / name


def move_to_archive(path: Path) -> None:
    if not path.exists():
        return
    ARCHIVE_DIR.mkdir(exist_ok=True)
    target = archive_path(path)
    shutil.move(str(path), str(target))
    logger.info("archive로 이동: %s -> %s", path, target)


def load_done_ids(pages_file: Path) -> set[int]:
    if not pages_file.exists():
        return set()
    done = set()
    with pages_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("pages.jsonl %d번째 줄 무시: %s", line_no, e)
    return done


def append_record(pages_file: Path, record: PageRecord) -> None:
    with pages_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 실행


def make_record(
    item: TocItem, html: str, cache_file: Path, by_id: dict[int, TocItem]
) -> PageRecord:
    text = clean_text(extract_body(html))
    fetched_at = datetime.fromtimestamp(cache_file.stat().st_mtime)  # noqa: DTZ006
    return PageRecord(
        id=item.id,
        parent_id=item.parent_id,
        depth=item.depth,
        order=item.order,
        title=item.title,
        breadcrumb=build_breadcrumb(item, by_id),
        url=f"{BASE_URL}/{item.id}",
        text=text,
        fetched_at=fetched_at.isoformat(timespec="seconds"),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def run(
    book_id: int,
    delay: float,
    user_agent: str,
    offline: bool = False,
    refresh: bool = False,
    limit: int | None = None,
) -> None:
    paths = Paths.for_book(book_id)
    if refresh:
        move_to_archive(paths.pages_file)
        move_to_archive(paths.failed_file)
        if not offline:
            move_to_archive(paths.raw_dir)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    headers = build_headers(user_agent)

    toc_html, used_network = load_or_fetch(
        f"{BASE_URL}/book/{book_id}", paths.raw_dir / "toc.html", headers, offline
    )
    toc = parse_toc(toc_html)
    if not toc:
        raise ValueError("목차 항목을 찾지 못함")
    by_id = {item.id: item for item in toc}
    done = load_done_ids(paths.pages_file)
    targets = [item for item in toc if item.id not in done][:limit]
    logger.info(
        "목차 %d개, 완료 %d개, 이번 대상 %d개", len(toc), len(done), len(targets)
    )

    failed: dict[int, str] = {}
    for n, item in enumerate(targets, start=1):
        if used_network:
            time.sleep(delay)
        cache_file = paths.raw_dir / f"{item.id}.html"
        try:
            html, used_network = load_or_fetch(
                f"{BASE_URL}/{item.id}", cache_file, headers, offline
            )
            record = make_record(item, html, cache_file, by_id)
        except FetchBlockedError as e:
            logger.error("차단 응답으로 중단: %s", e)
            failed[item.id] = str(e)
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            UnicodeDecodeError,
            FileNotFoundError,
            ValueError,
        ) as e:
            logger.warning("[%d/%d] 실패 %s: %s", n, len(targets), item.id, e)
            failed[item.id] = str(e)
            used_network = False
            continue
        append_record(paths.pages_file, record)
        logger.info(
            "[%d/%d] %s %s (%d자)",
            n,
            len(targets),
            item.id,
            item.title,
            len(record.text),
        )

    # 지난 실행의 실패 기록은 이번 결과로 대체한다
    move_to_archive(paths.failed_file)
    if failed:
        paths.failed_file.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.warning("실패 %d건 -> %s", len(failed), paths.failed_file)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="위키독스 책 텍스트 수집·정제")
    parser.add_argument("--book-id", type=int, default=1)
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("WIKIDOCS_DELAY_SEC", DEFAULT_DELAY_SEC)),
        help="요청 간격(초)",
    )
    parser.add_argument("--offline", action="store_true", help="캐시만 사용")
    parser.add_argument(
        "--refresh", action="store_true", help="기존 결과를 archive로 옮기고 다시 생성"
    )
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 페이지 수")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(
        book_id=args.book_id,
        delay=args.delay,
        user_agent=os.getenv("WIKIDOCS_USER_AGENT", DEFAULT_USER_AGENT),
        offline=args.offline,
        refresh=args.refresh,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
