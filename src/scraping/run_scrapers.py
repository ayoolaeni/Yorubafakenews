"""CLI: python -m src.scraping.run_scrapers --source all --class genuine --limit 500

Invoked by `make corpus`. Each source's articles land in
data/raw/<class>/<source_slug>.jsonl, one JSON record per line; reruns
resume rather than re-scraping URLs already on disk.
"""
from __future__ import annotations

import argparse

from src.config import settings
from src.scraping.base_scraper import run_source, slugify
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="all", help="Source name to scrape, or 'all'.")
    parser.add_argument("--class", dest="label", required=True, choices=["genuine", "fake"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to hold per source (existing + new); omit for no cap.",
    )
    args = parser.parse_args()

    sources = settings.sources.get(args.label)
    if not isinstance(sources, list):
        raise SystemExit(f"No scrapeable source list configured for --class {args.label!r}")

    if args.source != "all":
        wanted = args.source.lower()
        sources = [s for s in sources if s["name"].lower() == wanted or slugify(s["name"]) == wanted]
        if not sources:
            raise SystemExit(f"No source named {args.source!r} configured for --class {args.label!r}")

    defaults = settings.sources.get("scraper_defaults", {})
    total_added = 0
    for source_cfg in sources:
        total_added += run_source(
            source_cfg, args.label, args.limit, settings.contact_email, settings.paths.raw, defaults
        )

    logger.info("done: class=%s sources=%d total new articles=%d", args.label, len(sources), total_added)


if __name__ == "__main__":
    main()