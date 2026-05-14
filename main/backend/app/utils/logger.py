"""Structured logging configuration."""
import logging
import sys

from app.config import settings


def setup_logger(name: str = "healthcare_dl") -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if settings.ENV == "production":
        # JSON-style format for production
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}'
        )
    else:
        # Readable format for development
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(module)-20s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger()
