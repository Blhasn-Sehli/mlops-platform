"""
Centralized structlog configuration.

All modules get a logger via:
    from src.logging_config import get_logger
    log = get_logger(__name__)

Output is JSON (machine-parseable in prod, readable in dev via `jq`).
"""

import logging
import sys

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging. Safe to call multiple times."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "mlflow", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    setup_logging()
    return structlog.get_logger(name)
