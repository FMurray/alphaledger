from alphaledger.config import settings
import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Optional


class LogHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console()

    def emit(self, record):
        level = record.levelname
        message = self.format(record)
        if level == "DEBUG":
            self.console.print(f"[blue]{message}[/blue]")
        elif level == "INFO":
            self.console.print(f"[green]{message}[/green]")
        elif level == "WARNING":
            self.console.print(f"[yellow]{message}[/yellow]")
        elif level == "ERROR":
            self.console.print(f"[red]{message}[/red]")


def configure_logging(
    level: Optional[str] = None, module_name: str = "alphaledger"
) -> logging.Logger:
    """
    Configure logging for AlphaLedger with enhanced formatting and color-coding.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - defaults to settings
        module_name: Name for the logger

    Returns:
        Configured logger instance
    """
    # Set up log level - use settings or param value, defaulting to INFO
    log_level = level or getattr(settings, "log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger with basic settings
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",  # Simple format as Rich handler adds its own formatting
        datefmt="[%X]",
        handlers=[
            LogHandler(
                rich_tracebacks=True,
                markup=True,  # Enable Rich markup in log messages
                show_time=True,
                console=Console(width=120),  # Control console width
                tracebacks_show_locals=True,  # Show local vars in tracebacks
                tracebacks_extra_lines=2,
                omit_repeated_times=False,
            )
        ],
    )

    # Get logger for the specified module
    logger = logging.getLogger(module_name)

    # Set level specifically for this logger
    logger.setLevel(numeric_level)

    return logger


console = Console()

# Create global logger instance
log = configure_logging()


def get_logger(module_name: str = None) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        module_name: Optional module name (defaults to 'alphaledger' if None)

    Returns:
        Configured logger for the module
    """
    if module_name is None:
        return log

    return logging.getLogger(module_name)


def main() -> None:
    """Main entry point for alphaledger."""
    log.info("Hello from [bold green]alphaledger[/bold green]!")


def build_kb() -> None:
    """
    Build the knowledge base from a specified universe.

    This is an entry point that delegates to the knowledge_base module.
    """
    from alphaledger.knowledge_base import build_kb as kb_build

    if settings.verbose:
        log.info(
            f"Building knowledge base from [bold blue]{settings.universe_name}[/bold blue]..."
        )

    # Call the implementation in the knowledge_base module
    kb_build(log)
