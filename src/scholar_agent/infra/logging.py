import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | None = None,
    log_filename: str = "scholar_agent.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure logging to both console and a rotating file.

    Args:
        level: Logging level (default INFO).
        log_dir: Directory for log files. Defaults to ``logs/`` relative to
            the project root (detected via pyproject.toml) or CWD.
        log_filename: Name of the log file inside *log_dir*.
        max_bytes: Max size per log file before rotation (default 10 MB).
        backup_count: Number of rotated backups to keep (default 3).
    """
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Resolve log directory
    if log_dir is None:
        log_dir = _resolve_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when called multiple times
    if not root.handlers:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(fmt))
        root.addHandler(console)

    file_path = os.path.join(log_dir, log_filename)
    # Only add file handler if not already present
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(file_path)
        for h in root.handlers
    )
    if not has_file_handler:
        file_handler = RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(file_handler)


def _resolve_log_dir() -> str:
    """Find the project root (directory containing pyproject.toml) and return logs/ under it."""
    here = os.path.abspath(os.path.dirname(__file__))
    # Walk up from src/scholar_agent/infra/ to find pyproject.toml
    current = here
    for _ in range(6):
        if os.path.isfile(os.path.join(current, "pyproject.toml")):
            return os.path.join(current, "logs")
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback: logs/ relative to CWD
    return os.path.join(os.getcwd(), "logs")
