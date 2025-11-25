import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests

BASE_URL = "https://rollcall.com/wp-json/factbase/v1/twitter"

BASE_PARAMS: Dict[str, Any] = {
    "platform": "truth social",
    "sort": "date",
    "sort_order": "desc",
    "dateFilter": "last_year",
    "format": "json",
}

START_DATE_UTC = datetime(2025, 1, 20, tzinfo=timezone.utc)
SLEEP_SECONDS = 5

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "trump_truthsocial_since_2025-01-20.jsonl",
)


def parse_post_date_to_utc(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_text(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    text = raw_text.strip()
    if not text or text in ("[Video]", "[Image]"):
        return None
    return text


def _request_page(page: int) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    params = dict(BASE_PARAMS)
    params["page"] = page

    resp = requests.get(BASE_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")

    payload = resp.json()
    posts: List[Dict[str, Any]] = payload.get("data", []) or []
    meta: Dict[str, Any] = payload.get("meta", {}) or {}
    page_count = meta.get("page_count")
    return posts, page_count


def _iter_valid_records(posts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Возвращает (records, reached_start_date)."""
    records: List[Dict[str, Any]] = []

    for post in posts:
        date_str = post.get("date")
        if not date_str:
            continue

        try:
            post_dt_utc = parse_post_date_to_utc(date_str)
        except Exception:
            continue

        if post_dt_utc < START_DATE_UTC:
            return records, True

        text = normalize_text(post.get("text"))
        if text is None:
            continue

        records.append({"date": date_str, "text": text})

    return records, False


def fetch_and_save_trump_posts() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Writing to: {OUTPUT_PATH}")

    page = 1
    reached_start_date = False
    last_page: Optional[int] = None

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        while True:
            print(f"[INFO] Requesting page {page} ...")

            try:
                posts, last_page = _request_page(page)
            except RuntimeError as e:
                print(f"[WARN] {e} for page {page}, stopping.")
                break

            if not posts:
                print(f"[INFO] Empty data for page {page}, stopping.")
                break

            records, reached_start_date = _iter_valid_records(posts)

            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if reached_start_date:
                print(
                    f"[INFO] Reached posts older than {START_DATE_UTC.isoformat()}, "
                    f"cutting off at page {page}."
                )
                break

            if last_page is not None and page >= last_page:
                print(f"[INFO] Reached last page {last_page}, stopping.")
                break

            page += 1
            print(f"[SLEEP] {SLEEP_SECONDS} seconds before next page...")
            time.sleep(SLEEP_SECONDS)

    print("[DONE] Finished fetching posts.")


if __name__ == "__main__":
    fetch_and_save_trump_posts()
