import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

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
    if not text:
        return None
    if text in ("[Video]", "[Image]"):
        return None
    return text


def fetch_and_save_trump_posts() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    current_page = 1
    reached_start_date = False

    print(f"Writing to: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        while True:
            params = dict(BASE_PARAMS)
            params["page"] = current_page

            print(f"[INFO] Requesting page {current_page} ...")
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"[WARN] HTTP {resp.status_code} for page {current_page}, stopping.")
                break

            payload = resp.json()
            posts: List[Dict[str, Any]] = payload.get("data", [])
            meta: Dict[str, Any] = payload.get("meta", {}) or {}
            page_count = meta.get("page_count")

            if not posts:
                print(f"[INFO] Empty data for page {current_page}, stopping.")
                break

            for post in posts:
                date_str = post.get("date")
                text_raw = post.get("text")

                if not date_str:
                    continue

                try:
                    post_dt_utc = parse_post_date_to_utc(date_str)
                except Exception as e:
                    print(f"[WARN] Failed to parse date '{date_str}': {e}")
                    continue

                if post_dt_utc < START_DATE_UTC:
                    reached_start_date = True
                    print(
                        f"[INFO] Reached posts older than {START_DATE_UTC.isoformat()}, "
                        f"cutting off at page {current_page}."
                    )
                    break

                text = normalize_text(text_raw)
                if text is None:
                    continue

                record = {
                    "date": date_str,
                    "text": text,
                }
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")

            if reached_start_date:
                break

            if page_count is not None and current_page >= page_count:
                print(f"[INFO] Reached last page {page_count}, stopping.")
                break

            current_page += 1
            print(f"[SLEEP] {SLEEP_SECONDS} seconds before next page...")
            time.sleep(SLEEP_SECONDS)

    print("[DONE] Finished fetching posts.")


if __name__ == "__main__":
    fetch_and_save_trump_posts()
