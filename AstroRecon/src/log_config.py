"""Centralized logging configuration for the AstroRecon project.

Usage (in any module):
    import logging
    logger = logging.getLogger(__name__)

Call ``setup_logging()`` once from the CLI entry-point (train.py) before any
log statements execute.  Every other module simply does ``getLogger(__name__)``
and inherits the format, level, and handlers configured here.
"""

import logging
import sys
from pathlib import Path


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File log path (created lazily)
_LOG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "logs"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: str = "astrorecon.log",
) -> None:
    """Configure the root logger with console + optional rotating file handler.

    Parameters
    ----------
    level : int
        Minimum severity (default ``logging.INFO``).
    log_to_file : bool
        If *True*, also write to ``outputs/logs/<log_filename>``.
    log_filename : str
        Name of the log file inside ``outputs/logs/``.
    """
    root = logging.getLogger()

    # Avoid duplicate handlers if called more than once
    if root.handlers:
        return

    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ---- Console handler (stderr) ----
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ---- File handler (optional) ----
    if log_to_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            _LOG_DIR / log_filename,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quieten overly chatty libraries.  Only touch loggers that already
    # exist: ``logging.getLogger(name)`` caches the name with the current
    # logger class, and creating "astropy" here before astropy imports makes
    # its own ``_init_log()`` receive a plain Logger lacking ``_set_defaults``
    # (AttributeError at import).
    for lib in ("astroquery", "astropy", "urllib3", "matplotlib"):
        if lib in logging.Logger.manager.loggerDict:
            logging.getLogger(lib).setLevel(logging.WARNING)

