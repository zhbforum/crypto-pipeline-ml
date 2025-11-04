import logging
import os

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s [%(levelname)s] %(message)s"

root = logging.getLogger()
if not root.handlers:
    logging.basicConfig(level=_LEVEL, format=_FMT)

def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(_LEVEL)
    return lg
