import asyncio
import math
import time
from app.constants import _INTERVAL_MAP_SECONDS


def interval_seconds(interval: str) -> int:
    if interval not in _INTERVAL_MAP_SECONDS:
        raise ValueError(f"Unsupported INTERVAL: {interval}")
    return _INTERVAL_MAP_SECONDS[interval]


async def sleep_until_next_boundary(period_sec: int) -> None:
    now = time.time()
    next_edge = (math.floor(now / period_sec) + 1) * period_sec
    await asyncio.sleep(max(0.0, next_edge - now + 0.5))
