#!/usr/bin/env python3
"""Shared structlog configuration.

Call configure_logging() once at CLI entry point before any logging.
Use get_logger() in all modules instead of logging.getLogger().

Timestamps, log levels, and module names are injected by the shared
processor pipeline — no module configures its own formatting.
"""

import logging
import sys

import structlog


def configure_logging(verbose: bool = False, json_output: bool = False) -> None:
    """Configure the shared structlog processor pipeline.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
        json_output: If True, render as JSON. Otherwise human-readable console output.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            timestamp_key="timestamp",
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Suppress noisy third-party loggers
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("htmldate").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Bound structlog logger.
    """
    return structlog.get_logger(name)
