"""
Logger Utility Module - Centralized logging configuration.

This module provides a configured logging system for the entire application.
It supports file and console output, log rotation, and structured logging.

Features:
    - Colored console output for better readability
    - File-based logging with rotation
    - JSON structured logging option
    - Per-module logger configuration
    - Context-aware logging (adds correlation IDs)

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Application started")

TODO: Add log aggregation support (ELK, Loki)
TODO: Add performance metrics logging
TODO: Add sensitive data masking
TODO: Add async logging for high-throughput scenarios
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Optional: colorama for Windows color support
try:
    import colorama
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


# =============================================================================
# Color Codes for Console Output
# =============================================================================

class LogColors:
    """ANSI color codes for log levels."""
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Log level colors
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[41m"   # Red background
    
    # Component colors
    TIMESTAMP = "\033[90m"  # Gray
    NAME = "\033[35m"       # Magenta
    MESSAGE = "\033[37m"    # White


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log output.
    
    Only applies colors when outputting to a TTY (terminal).
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }
    
    def __init__(self, fmt: str, datefmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        if not self.use_colors:
            return super().format(record)
        
        # Add color to level name
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        record.levelname = f"{level_color}{record.levelname:8}{LogColors.RESET}"
        
        # Add color to logger name
        record.name = f"{LogColors.NAME}{record.name}{LogColors.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        # Add timestamp color
        if self.datefmt:
            timestamp = datetime.now().strftime(self.datefmt)
            formatted = formatted.replace(
                timestamp, 
                f"{LogColors.TIMESTAMP}{timestamp}{LogColors.RESET}",
                1
            )
        
        return formatted


class ContextFilter(logging.Filter):
    """
    Filter that adds context information to log records.
    
    Adds correlation_id and other context for request tracking.
    
    TODO: Integrate with async context for correlation IDs
    """
    
    def __init__(self):
        super().__init__()
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set context values that will be added to all log records."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context values."""
        self._context.clear()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the log record."""
        for key, value in self._context.items():
            setattr(record, key, value)
        
        # Set defaults for missing context
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = '-'
        
        return True


# =============================================================================
# Logger Configuration
# =============================================================================

# Global context filter for correlation IDs
_context_filter = ContextFilter()

# Track configured loggers
_configured_loggers: set = set()

# Default format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    rotation: str = "daily",
    max_size_mb: int = 10,
    backup_count: int = 7,
    json_format: bool = False,
) -> None:
    """
    Configure the root logger and all handlers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = console only)
        log_format: Log message format
        date_format: Date format string
        rotation: Rotation strategy ('daily', 'size', 'none')
        max_size_mb: Max file size before rotation (for size rotation)
        backup_count: Number of backup files to keep
        json_format: Whether to use JSON structured logging
    
    TODO: Add JSON formatter implementation
    """
    log_format = log_format or DEFAULT_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add context filter
    root_logger.addFilter(_context_filter)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(log_format, date_format, use_colors=True)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        _add_file_handler(
            root_logger,
            log_file,
            log_format,
            date_format,
            rotation,
            max_size_mb,
            backup_count,
        )
    
    # Mark as configured
    _configured_loggers.add('root')


def _add_file_handler(
    logger: logging.Logger,
    log_file: str,
    log_format: str,
    date_format: str,
    rotation: str,
    max_size_mb: int,
    backup_count: int,
) -> None:
    """Add a file handler to the logger."""
    # Create log directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create appropriate handler based on rotation strategy
    if rotation == "daily":
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
    elif rotation == "size":
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
    
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured Logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Module initialized")
    """
    # Auto-configure if not done yet
    if 'root' not in _configured_loggers:
        configure_logging()
    
    logger = logging.getLogger(name)
    
    # Add context filter if not already added
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(_context_filter)
    
    return logger


def set_log_context(**kwargs) -> None:
    """
    Set context values that will be added to all log records.
    
    Useful for adding correlation IDs, user IDs, etc.
    
    Args:
        **kwargs: Key-value pairs to add to log context
    
    Example:
        set_log_context(correlation_id="abc-123", user_id="user-456")
    """
    _context_filter.set_context(**kwargs)


def clear_log_context() -> None:
    """Clear all log context values."""
    _context_filter.clear_context()


# =============================================================================
# Convenience Functions
# =============================================================================

def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Context message
        exc: The exception to log
        level: Log level to use
    """
    logger.log(level, f"{message}: {type(exc).__name__}: {exc}", exc_info=True)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    threshold_ms: float = 1000,
) -> None:
    """
    Log operation performance.
    
    Logs as DEBUG normally, WARNING if over threshold.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        threshold_ms: Warning threshold in milliseconds
    """
    level = logging.WARNING if duration_ms > threshold_ms else logging.DEBUG
    logger.log(level, f"Performance: {operation} took {duration_ms:.2f}ms")


# =============================================================================
# Module Initialization
# =============================================================================

def init_from_config(config: Dict[str, Any]) -> None:
    """
    Initialize logging from configuration dictionary.
    
    Expected config structure matches settings.yaml 'general' section.
    
    Args:
        config: Configuration dictionary
    """
    configure_logging(
        level=config.get("log_level", "INFO"),
        log_file=config.get("log_file"),
        log_format=config.get("log_format"),
        rotation=config.get("log_rotation", "daily"),
        max_size_mb=config.get("log_max_size_mb", 10),
        backup_count=config.get("log_backup_count", 7),
    )


# Pre-configure with defaults when module is imported
# This ensures logging works even if configure_logging is never called
if 'root' not in _configured_loggers:
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
