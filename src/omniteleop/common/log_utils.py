#!/usr/bin/env python3
"""Utility functions for managing logging output."""

import sys
import io
from contextlib import contextmanager

@contextmanager
def suppress_loguru_module(module_name: str = "dexmotion", enabled: bool = True):
    """Context manager to suppress loguru logs from a specific module.

    This is useful for suppressing verbose output from third-party libraries
    that use loguru for logging, especially when they interfere with
    interactive displays.

    Args:
        module_name: Name of the module to suppress logs from.
                    Defaults to "dexmotion" for backwards compatibility.
        enabled: If True, suppression is active. If False, logs pass through normally.
                Defaults to True for backwards compatibility.

    Example:
        with suppress_loguru_module("dexmotion"):
            # Code that calls dexmotion functions
            motion_manager.ik(...)  # No verbose logs will be printed

        with suppress_loguru_module("dexmotion", enabled=False):
            # Suppression disabled, logs will be shown
            motion_manager.ik(...)  # Logs will be printed
    """
    from loguru import logger as loguru_logger

    if not enabled:
        # If suppression is disabled, just yield without doing anything
        yield
        return

    try:
        # Temporarily disable loguru messages from the specified module
        loguru_logger.disable(module_name)

        # Also redirect stdout/stderr to suppress any direct prints
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        yield
    finally:
        # Re-enable logging for the module
        loguru_logger.enable(module_name)

        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Convenience function for the most common use case
def suppress_dexmotion_logs():
    return suppress_loguru_module("dexmotion")
