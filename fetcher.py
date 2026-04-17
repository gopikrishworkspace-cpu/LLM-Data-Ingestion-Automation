"""
fetcher.py — Pre-pipeline input resolver.

Resolves an input path to a clean plain-text file ready for chunking:

  - If the file contains a single URL → fetch the page, strip HTML, save
    a .txt beside the original so the pipeline has real content.
  - If the file is already multi-line text → return path unchanged.
  - Automatically skips re-fetching if the resolved file already exists
    and is newer than the source.

No external dependencies — uses only Python stdlib.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML → plain-text stripper
# ---------------------------------------------------------------------------

# Tags whose full content (including children) we discard
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "button", "svg", "img",
    "meta", "link", "head",
})

# Block-level tags that should emit a blank line after their content
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "br", "hr",
    "blockquote", "pre", "code",
})


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._current_tag = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        self._current_tag = tag
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped + " ")

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace runs, normalise newlines
        raw = re.sub(r" {2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"^(https?://[^\s]+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_url(text: str) -> str | None:
    """Return the single URL if the file contains exactly one, else None."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) == 1 and _URL_RE.match(lines[0]):
        return lines[0]
    # Also handle multi-line files that are just a URL on the first non-blank line
    if len(lines) <= 3:
        for line in lines:
            if _URL_RE.match(line):
                return line
    return None


# ---------------------------------------------------------------------------
# HTTP fetch with retries
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_url(url: str, timeout: int = 30, max_retries: int = 3) -> str:
    """Fetch a URL and return the raw HTML as a string."""
    req = urllib.request.Request(url, headers=_HEADERS, method="GET")
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                charset = "utf-8"
                content_type = resp.headers.get("Content-Type", "")
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                raw = resp.read()
                return raw.decode(charset, errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
            logger.warning("Fetch attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MIN_USEFUL_CHARS = 500  # below this, fetched content is considered useless


def resolve_input(input_path: str | Path) -> Path:
    """Given an input file path, return a path to a file with real text content.

    If the input file contains a URL:
      - Fetch the page, convert to text
      - Save as <original_stem>_fetched.txt next to the original
      - Return the path to the fetched file

    If the input already has substantial text content, return it unchanged.
    If fetch fails, return the original path (pipeline will process the URL
    text itself, same as before).
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    raw = src.read_text(encoding="utf-8", errors="replace").strip()

    url = _extract_url(raw)
    if url is None:
        logger.info("Input %s looks like a document (not a URL) — using as-is", src.name)
        return src

    # Check if we already have a cached fetch
    fetched_path = src.with_name(src.stem + "_fetched.txt")
    if fetched_path.exists() and fetched_path.stat().st_mtime > src.stat().st_mtime:
        logger.info("Using cached fetched content: %s", fetched_path.name)
        return fetched_path

    logger.info("Input is a URL — fetching: %s", url)
    try:
        html = fetch_url(url)
    except RuntimeError as exc:
        logger.error("Fetch failed, falling back to URL text: %s", exc)
        return src

    text = html_to_text(html)

    if len(text) < _MIN_USEFUL_CHARS:
        logger.warning(
            "Fetched content too short (%d chars) — page likely requires JavaScript "
            "(client-side rendered).\n"
            "To ingest this page manually:\n"
            "  1. Open the URL in a browser\n"
            "  2. Select All → Copy the visible text\n"
            "  3. Paste into:  input/%s_content.txt\n"
            "  4. Run:  python run.py input/%s_content.txt\n"
            "Falling back to URL text (pipeline will likely hallucinate entities).",
            len(text), src.stem, src.stem,
        )
        return src

    fetched_path.write_text(text, encoding="utf-8")
    logger.info(
        "Fetched %d chars from %s → saved to %s",
        len(text), url, fetched_path.name,
    )
    return fetched_path
