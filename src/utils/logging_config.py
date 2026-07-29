"""Consistent logging setup for scripts and modules.

Use ``get_logger(__name__)`` rather than ``print`` so scraper/training runs
produce a filterable, timestamped log (needed to verify e.g. "test set
opened exactly once" from run logs, per the anti-leakage rules).
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
