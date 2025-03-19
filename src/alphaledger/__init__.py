from alphaledger.config import settings


def configure_logging():
    """Configure logging for AlphaLedger."""
    import logging
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    return logging.getLogger("alphaledger")


log = configure_logging()


def main() -> None:
    """Main entry point for alphaledger."""
    print("Hello from alphaledger!")


def build_kb() -> None:
    """
    Build the knowledge base from a specified universe.

    This is an entry point that delegates to the knowledge_base module.
    """
    from alphaledger.knowledge_base import build_kb as kb_build

    if settings.verbose:
        log.info(f"Building knowledge base from {settings.universe_name}...")

    # Call the implementation in the knowledge_base module
    kb_build(log)
