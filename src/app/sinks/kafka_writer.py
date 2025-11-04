from __future__ import annotations
import json
import time
from typing import Any, Dict, Iterable, Tuple
from confluent_kafka import Producer, KafkaException, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from app.constants import KAFKA_CONFIG


def _ensure_topic(conf: Dict[str, str], topic: str, partitions: int = 3, replication: int = 3, timeout_s: int = 30) -> None:
    """Создаёт Kafka-топик, если его ещё нет."""
    admin_conf = {
        "bootstrap.servers": conf["bootstrap.servers"],
        "security.protocol": conf.get("security.protocol", "SASL_SSL"),
        "sasl.mechanisms": conf.get("sasl.mechanisms", "PLAIN"),
        "sasl.username": conf["sasl.username"],
        "sasl.password": conf["sasl.password"],
    }
    admin = AdminClient(admin_conf)
    md = admin.list_topics(timeout=10)
    if topic in md.topics and md.topics[topic].error is None:
        return

    fs = admin.create_topics([NewTopic(topic, num_partitions=partitions, replication_factor=replication)])
    f = fs[topic]
    try:
        f.result(timeout=timeout_s)
        print(f"[kafka] topic created: {topic}")
    except Exception as e:
        if "TopicAlreadyExistsError" not in str(e):
            print(f"[kafka] topic create warning: {e}")


class KafkaWriter:
    """Kafka-продюсер, использующий конфигурацию из .env через constants.KAFKA_CONFIG."""

    def __init__(self, config: Dict[str, str], topic: str):
        self._conf = dict(config)
        self._topic = topic
        self._conf.setdefault("acks", "all")
        self._conf.setdefault("client.id", self._conf.get("client.id", "crypto-pipeline"))

        print(f"[kafka] bootstrap={self._conf.get('bootstrap.servers')} topic={topic}")
        try:
            _ensure_topic(self._conf, topic, partitions=3, replication=3)
        except Exception as e:
            print(f"[kafka] ensure_topic skipped: {e}")

        self._p = Producer(self._conf)
        self._delivered_ok = 0
        self._delivered_err = 0

    @classmethod
    def from_env(cls, topic: str) -> KafkaWriter:
        """Создание продюсера напрямую из .env"""
        return cls(KAFKA_CONFIG, topic)

    def _on_delivery(self, err, _msg):
        if err:
            self._delivered_err += 1
            print(f"[kafka] delivery failed: {err}")
        else:
            self._delivered_ok += 1

    def send_batch(self, rows: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
        """Отправляет список сообщений в Kafka."""
        self._delivered_ok = 0
        self._delivered_err = 0

        for r in rows:
            key = (r.get("symbol") or r.get("event_type") or "event")
            payload = json.dumps(r).encode("utf-8")
            while True:
                try:
                    self._p.produce(self._topic, payload, key=str(key), callback=self._on_delivery)
                    break
                except BufferError:
                    self._p.poll(0.2)
                except KafkaException as e:
                    if (
                        e.args
                        and isinstance(e.args[0], KafkaError)
                        and e.args[0].code() == KafkaError.UNKNOWN_TOPIC_OR_PART
                    ):
                        raise
                    time.sleep(0.2)

            self._p.poll(0)

        self._p.flush(10.0)
        return self._delivered_ok, self._delivered_err
