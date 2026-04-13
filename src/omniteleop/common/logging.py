#!/usr/bin/env python3
"""Logging configuration for the teleoperation system."""

import sys
from loguru import logger

def setup_logging(debug: bool = False):
    """Configure loguru for efficient logging.

    Args:
        debug: Enable debug level logging
    """
    # Remove default handler
    logger.remove()

    # Add custom handler with simplified format
    if debug:
        # Debug mode: minimal format for high-frequency output
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <5}</level> | <cyan>{message}</cyan>",
            level="DEBUG",
            colorize=True,  # Enable color markup parsing
            enqueue=False,  # Synchronous logging for Rich compatibility
            backtrace=False,
            diagnose=False,
        )
    else:
        # Normal mode: more detailed format
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}",
            level="INFO",
            colorize=True,  # Enable color markup parsing
            enqueue=False,  # Synchronous logging for Rich compatibility
            backtrace=True,
            diagnose=True,
        )

    return logger
