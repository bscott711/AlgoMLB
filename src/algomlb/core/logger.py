import os
import sys
from pathlib import Path
from loguru import logger as _logger

# Logic:
# 1. Human-friendly stderr output (Rich, no JSONL).
# 2. Machine-friendly file output (JSONL, rotating, compressed).

LOG_DIR = Path(os.getenv("ALGOMLB_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove the default loguru sink
_logger.remove()

# human-friendly stderr sink
_logger.add(
    sys.stderr,
    level=os.getenv("ALGOMLB_LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    colorize=True,
    # Filter out agent mode output from the human log if desired
    filter=lambda r: not r["extra"].get("agent_mode", False),
)

# machine-friendly file sink (JSONL)
_logger.add(
    LOG_DIR / "algomlb.log",
    level="DEBUG",
    serialize=True,  # This produces JSON lines
    rotation="20 MB",
    retention="14 days",
    compression="gz",
    enqueue=True,  # Multiprocess safe
)

# re-export as the standard logger for use everywhere
logger = _logger.bind()
