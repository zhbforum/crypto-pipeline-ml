from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable
import json
from confluent_kafka import Producer  


def _load_properties(path: Path) -> Dict[str, str]:
    props: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Kafka properties file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        props[k.strip()] = v.strip()
    return props


class KafkaWriter:
    def __init__(self, producer: Producer, topic: str):
        self._p = producer
        self._topic = topic

    @classmethod
    def from_properties(cls, properties_path: str | Path, topic: str) -> "KafkaWriter":
        conf = _load_properties(Path(properties_path))
        conf.setdefault("client.id", "crypto-pipeline")
        prod = Producer(conf)
        return cls(prod, topic)

    def send_batch(self, rows: Iterable[Dict[str, Any]]) -> int:
        sent = 0
        for r in rows:
            key = (r.get("symbol") or r.get("event_type") or "event")
            self._p.produce(self._topic, json.dumps(r).encode("utf-8"), key=str(key))
            self._p.poll(0)
            sent += 1
        self._p.flush(10.0)
        return sent
