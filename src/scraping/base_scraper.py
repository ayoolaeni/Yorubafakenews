"""Fetching, throttling, and HTML/RSS extraction shared by every source.

Selectors live in ``config/sources.yaml`` rather than here (see the comment
at the top of that file) so a broken selector is a config edit, not a code
change. ``run_source`` is resumable: it reads whatever is already in the
per-source ``.jsonl`` output, skips URLs already collected, and stops once
``limit`` new+existing records are on disk.
"""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from src.utils.io import append_jsonl, read_jsonl
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Throttle:
    """Enforces a minimum interval between requests to a single source."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval = min_interval_seconds
        self._last_request: float | None = None

    def wait(self) -> None:
        if self._last_request is not None:
            remaining = self.min_interval - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


def build_session(contact_email: str, user_agent_template: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": user_agent_template.format(contact_email=contact_email)}
    )
    return session


def fetch(
    url: str,
    session: requests.Session,
    throttle: Throttle,
    timeout_seconds: float,
    max_retries: int,
    backoff_base_seconds: float,
) -> str | None:
    for attempt in range(1, max_retries + 1):
        throttle.wait()
        try:
            resp = session.get(url, timeout=timeout_seconds)
            if resp.status_code == 200:
                return resp.text
            logger.warning(
                "%s -> HTTP %s (attempt %d/%d)", url, resp.status_code, attempt, max_retries
            )
        except requests.RequestException as exc:
            logger.warning("%s -> %s (attempt %d/%d)", url, exc, attempt, max_retries)
        if attempt < max_retries:
            time.sleep(backoff_base_seconds**attempt)
    logger.error("giving up on %s after %d attempts", url, max_retries)
    return None


def extract_links(html: str, base_url: str, selector: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.select(selector):
        href = tag.get("href")
        if not href:
            continue
        absolute = urljoin(base_url, href)
        if urlparse(absolute).scheme not in ("http", "https"):
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def listing_page_url(base_url: str, page: int) -> str:
    """WordPress-style pagination: base_url, then base_url/page/2/, /page/3/, ..."""
    if page <= 1:
        return base_url
    root = base_url if base_url.endswith("/") else base_url + "/"
    return urljoin(root, f"page/{page}/")


def collect_listing_links(
    source_cfg: dict[str, Any],
    session: requests.Session,
    throttle: Throttle,
    timeout_seconds: float,
    max_retries: int,
    backoff_base_seconds: float,
) -> list[str]:
    """Fetches source_cfg["max_listing_pages"] (default 1) listing pages and
    merges their links, stopping early once a page yields nothing new --
    most sites' pagination eventually loops back or 404s past the last page.
    """
    max_pages = source_cfg.get("max_listing_pages", 1)
    seen: set[str] = set()
    links: list[str] = []
    for page in range(1, max_pages + 1):
        url = listing_page_url(source_cfg["base_url"], page)
        html = fetch(url, session, throttle, timeout_seconds, max_retries, backoff_base_seconds)
        if html is None:
            break
        page_links = extract_links(html, source_cfg["base_url"], source_cfg["listing_selector"])
        new_on_page = [link for link in page_links if link not in seen]
        if not new_on_page:
            break
        for link in new_on_page:
            seen.add(link)
            links.append(link)
    return links


def parse_rss_links(
    rss_url: str,
    session: requests.Session,
    throttle: Throttle,
    timeout_seconds: float,
    max_retries: int,
    backoff_base_seconds: float,
) -> list[str]:
    text = fetch(rss_url, session, throttle, timeout_seconds, max_retries, backoff_base_seconds)
    if text is None:
        return []
    feed = feedparser.parse(text)
    return [entry.link for entry in feed.entries if getattr(entry, "link", None)]


def _text(node: Any) -> str:
    return node.get_text(" ", strip=True) if node is not None else ""


def scrape_genuine_article(url: str, html: str, source_cfg: dict[str, Any]) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    scope = soup.select_one(source_cfg["article_selector"]) if source_cfg.get("article_selector") else soup
    if scope is None:
        scope = soup

    title = _text(scope.select_one(source_cfg["title_selector"]))
    body = "\n".join(
        t for t in (_text(n) for n in scope.select(source_cfg["body_selector"])) if t
    )
    if not body:
        return None

    return {
        "url": url,
        "source": source_cfg["name"],
        "label": "genuine",
        "title": title,
        "text": body,
        "collected_at": now_iso(),
    }


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def find_claim_paragraph(soup: BeautifulSoup, claim_selector: str, claim_keyword: str | None) -> str:
    """Returns the text of the first candidate paragraph whose diacritic-
    stripped, lowercased text contains claim_keyword (also normalised the
    same way) -- e.g. matches both "Aheso:" and "Àhèsọ:" for keyword="aheso".
    Without a keyword, falls back to the first candidate (e.g. a real
    <blockquote> selector, where presence alone is the signal).
    """
    candidates = soup.select(claim_selector)
    if not claim_keyword:
        return _text(candidates[0]) if candidates else ""

    keyword = _strip_diacritics(claim_keyword).lower()
    for node in candidates:
        text = _text(node)
        if keyword in _strip_diacritics(text).lower():
            return text
    return ""


def scrape_fake_article(url: str, html: str, source_cfg: dict[str, Any]) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    claim = find_claim_paragraph(soup, source_cfg["claim_selector"], source_cfg.get("claim_keyword"))
    if not claim:
        return None

    verdict = "\n".join(
        t for t in (_text(n) for n in soup.select(source_cfg.get("verdict_selector", ""))) if t
    )
    title = soup.title.get_text(strip=True) if soup.title else ""

    return {
        "url": url,
        "source": source_cfg["name"],
        "label": "fake",
        "title": title,
        "text": claim,
        "verdict_text": verdict,
        "collected_at": now_iso(),
    }


def scrape_article(url: str, html: str, source_cfg: dict[str, Any], label: str) -> dict[str, Any] | None:
    if label == "genuine":
        return scrape_genuine_article(url, html, source_cfg)
    if label == "fake":
        return scrape_fake_article(url, html, source_cfg)
    raise ValueError(f"unknown label: {label!r}")


def run_source(
    source_cfg: dict[str, Any],
    label: str,
    limit: int | None,
    contact_email: str,
    output_dir: Path,
    defaults: dict[str, Any],
) -> int:
    name = source_cfg["name"]
    out_path = Path(output_dir) / label / f"{slugify(name)}.jsonl"

    existing = read_jsonl(out_path)
    seen_urls = {record["url"] for record in existing}
    collected = len(existing)
    if limit is not None and collected >= limit:
        logger.info("%s: already have %d/%d, skipping", name, collected, limit)
        return 0

    rate_limit = source_cfg.get("rate_limit_seconds", defaults["default_rate_limit_seconds"])
    throttle = Throttle(rate_limit)
    session = build_session(contact_email, defaults["user_agent_template"])
    timeout_seconds = defaults["timeout_seconds"]
    max_retries = defaults["max_retries"]
    backoff_base_seconds = defaults["backoff_base_seconds"]

    links: list[str] = []
    if source_cfg.get("type") == "rss_then_scrape" and source_cfg.get("rss"):
        links = parse_rss_links(
            source_cfg["rss"], session, throttle, timeout_seconds, max_retries, backoff_base_seconds
        )
    if not links:
        links = collect_listing_links(source_cfg, session, throttle, timeout_seconds, max_retries, backoff_base_seconds)

    new_links = [url for url in links if url not in seen_urls]
    logger.info("%s: found %d links (%d new)", name, len(links), len(new_links))

    added = 0
    for url in new_links:
        if limit is not None and collected >= limit:
            break
        html = fetch(url, session, throttle, timeout_seconds, max_retries, backoff_base_seconds)
        if html is None:
            continue
        record = scrape_article(url, html, source_cfg, label)
        if record is None:
            logger.warning("%s: no usable content extracted from %s", name, url)
            continue
        append_jsonl(out_path, record)
        seen_urls.add(url)
        collected += 1
        added += 1

    logger.info("%s: added %d new articles (total on disk: %d)", name, added, collected)
    return added
