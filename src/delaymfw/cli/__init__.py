import argparse
import logging


def str2bool(value: str) -> bool:
    """Parse booleans the way Julia's ArgParse does (``true``/``false``/``1``/``0``)."""
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
