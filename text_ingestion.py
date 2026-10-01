"""위키독스 책의 목차와 본문을 수집해 정제된 텍스트(JSONL)로 저장한다.

사용 예:
    python text_ingestion.py                 # 수집 (캐시가 있으면 재사용)
    python text_ingestion.py --offline       # 네트워크 없이 캐시만으로 재정제
    python text_ingestion.py --refresh       # 기존 결과를 archive로 옮기고 재수집
    python text_ingestion.py --url https://wikidocs.net/13   # 페이지 하나만
    python text_ingestion.py --url 13 20 --refresh           # 여러 페이지 다시 받기
    python text_ingestion.py --url https://wikidocs.net/book/1  # 책 전체
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
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

BASE_URL = "https://wikidocs.net"
ALLOWED_HOSTS = ("wikidocs.net", "www.wikidocs.net")
# 실행 위치와 상관없이 프로젝트 루트 기준으로 저장한다
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / ".data"
UNBOUND_DIR_NAME = "unbound"  # 책 목차에서 찾지 못한 페이지
ARCHIVE_DIR = PROJECT_ROOT / "archive"

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
PAGE_PATH = re.compile(r"^/(\d+)/?$")
BOOK_PATH = re.compile(r"^/book/(\d+)/?$")

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
    book_id: int | None
    book_title: str | None
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
    def for_book(cls, book_id: int | None) -> "Paths":
        name = f"book{book_id}" if book_id is not None else UNBOUND_DIR_NAME
        book_dir = DATA_ROOT / name
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


def find_cached_page(page_id: int) -> Path | None:
    """책을 모르는 상태에서 .data/*/raw/{id}.html 캐시를 찾는다."""
    return next(DATA_ROOT.glob(f"*/raw/{page_id}.html"), None)


# ---------------------------------------------------------------- 파싱


def parse_target(value: str) -> tuple[str, int]:
    """URL 또는 숫자를 ("page" | "book", id)로 해석한다."""
    value = value.strip()
    if value.isdigit():
        return "page", int(value)
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"위키독스 URL이 아님: {value}")
    if m := PAGE_PATH.match(parsed.path):
        return "page", int(m.group(1))
    if m := BOOK_PATH.match(parsed.path):
        return "book", int(m.group(1))
    raise ValueError(f"지원하지 않는 경로: {value}")


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def parse_book_info(html: str) -> tuple[int | None, str | None]:
    """목차 영역(#toc-data)에서 책 ID와 제목을 읽는다. 없으면 (None, None)."""
    soup = BeautifulSoup(html, "html.parser")
    toc_data = soup.select_one("#toc-data")
    if toc_data is None:
        return None, None
    title_tag = toc_data.select_one('a[href^="/book/"] strong')
    title = normalize(title_tag.get_text(strip=True)) if title_tag else None
    return _to_int(toc_data.get("data-book-id")), title


def parse_page_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one("h1.page-subject span.page-subject-text")
    return normalize(tag.get_text(strip=True)) if tag else None


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
    text = text.replace("\u00a0", " ")
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
    try:
        # .data/book1/raw/13.html -> book1_raw_13_{stamp}.html
        parts = path.relative_to(DATA_ROOT).parent.parts
    except ValueError:
        parts = (path.parent.name,)
    base = "_".join([*parts, path.stem])
    target = ARCHIVE_DIR / f"{base}_{stamp}{path.suffix}"
    # 같은 초에 여러 번 옮겨도 기존 archive를 덮어쓰지 않도록 번호를 붙인다
    n = 1
    while target.exists():
        target = ARCHIVE_DIR / f"{base}_{stamp}_{n}{path.suffix}"
        n += 1
    return target


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
    pages_file.parent.mkdir(parents=True, exist_ok=True)
    with pages_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _record_id(line: str) -> int | None:
    try:
        return json.loads(line)["id"]
    except (json.JSONDecodeError, KeyError):
        return None


def remove_records(pages_file: Path, ids: set[int]) -> None:
    """기존 파일은 archive로 옮기고, 해당 id를 뺀 나머지로 다시 쓴다."""
    lines = pages_file.read_text(encoding="utf-8").splitlines(keepends=True)
    keep = [line for line in lines if _record_id(line) not in ids]
    move_to_archive(pages_file)
    pages_file.write_text("".join(keep), encoding="utf-8")


# ---------------------------------------------------------------- 실행


def make_record(
    item: TocItem,
    html: str,
    cache_file: Path,
    by_id: dict[int, TocItem],
    book_id: int | None,
    book_title: str | None,
) -> PageRecord:
    text = clean_text(extract_body(html))
    fetched_at = datetime.fromtimestamp(cache_file.stat().st_mtime)  # noqa: DTZ006
    return PageRecord(
        id=item.id,
        book_id=book_id,
        book_title=book_title,
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
    _, book_title = parse_book_info(toc_html)
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
            record = make_record(item, html, cache_file, by_id, book_id, book_title)
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


def ingest_page(
    page_id: int, headers: dict[str, str], offline: bool, refresh: bool
) -> bool:
    """책 정보 없이 페이지 하나를 처리한다. 네트워크를 썼으면 True."""
    cached = find_cached_page(page_id)
    if cached and refresh and not offline:
        move_to_archive(cached)
        cached = None
    if cached:
        html, used_network = cached.read_text(encoding="utf-8"), False
    elif offline:
        raise FileNotFoundError(f"캐시 없음 (offline): {page_id}")
    else:
        html, used_network = fetch_html(f"{BASE_URL}/{page_id}", headers), True

    # 본문 페이지에 책 ID와 책 전체 목차가 함께 들어 있다
    book_id, book_title = parse_book_info(html)
    by_id = {item.id: item for item in parse_toc(html)}
    item = by_id.get(page_id)
    if item is None:
        logger.warning("%s: 책 목차에서 찾지 못해 %s로 저장", page_id, UNBOUND_DIR_NAME)
        book_id, book_title = None, None
        title = parse_page_title(html) or str(page_id)
        item = TocItem(id=page_id, parent_id=None, depth=0, order=-1, title=title)

    paths = Paths.for_book(book_id)
    cache_file = paths.raw_dir / f"{page_id}.html"
    if used_network:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(html, encoding="utf-8")
    elif cached != cache_file:
        cache_file = cached  # 다른 위치의 캐시를 그대로 사용

    already_done = page_id in load_done_ids(paths.pages_file)
    if already_done and not refresh:
        logger.info("%s: 이미 수집됨, 건너뜀 (--refresh로 교체)", page_id)
        return used_network
    record = make_record(item, html, cache_file, by_id, book_id, book_title)
    if already_done:
        remove_records(paths.pages_file, {page_id})
    append_record(paths.pages_file, record)
    logger.info(
        "%s %s -> %s (%d자)",
        page_id,
        " > ".join(record.breadcrumb),
        paths.pages_file,
        len(record.text),
    )
    return used_network


def run_targets(
    values: list[str],
    delay: float,
    user_agent: str,
    offline: bool,
    refresh: bool,
    limit: int | None,
) -> None:
    """--url로 받은 페이지·책 URL(또는 페이지 번호)을 순서대로 처리한다."""
    headers = build_headers(user_agent)
    used_network = False
    for value in values:
        try:
            kind, target_id = parse_target(value)
        except ValueError as e:
            logger.error("%s", e)
            continue
        if kind == "book":
            run(target_id, delay, user_agent, offline, refresh, limit)
            used_network = not offline
            continue
        if used_network:
            time.sleep(delay)
        try:
            used_network = ingest_page(target_id, headers, offline, refresh)
        except FetchBlockedError as e:
            logger.error("차단 응답으로 중단: %s", e)
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            UnicodeDecodeError,
            FileNotFoundError,
            ValueError,
        ) as e:
            logger.warning("%s 실패: %s", value, e)
            used_network = False


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
    parser.add_argument(
        "--url",
        nargs="+",
        metavar="URL",
        help="페이지/책 URL 또는 페이지 번호 (지정하면 --book-id 무시)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    user_agent = os.getenv("WIKIDOCS_USER_AGENT", DEFAULT_USER_AGENT)
    if args.url:
        run_targets(
            args.url, args.delay, user_agent, args.offline, args.refresh, args.limit
        )
        return
    run(
        book_id=args.book_id,
        delay=args.delay,
        user_agent=user_agent,
        offline=args.offline,
        refresh=args.refresh,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
