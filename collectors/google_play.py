from google_play_scraper import reviews, Sort
import pandas as pd
from datetime import datetime, timezone
import time

APP_IDS = {
    "melon": "com.iloen.melon",
    "spotify": "com.spotify.music",
    "youtube_music": "com.google.android.apps.youtube.music",
}

APP_NAMES = {
    "melon": "멜론",
    "spotify": "Spotify",
    "youtube_music": "YouTube Music",
}

# Google Play API 한 번에 가져올 수 있는 최대 누적 건수 (무한루프 방지)
HARD_LIMIT = 5000


def _to_naive_datetime(dt):
    if dt is None:
        return None
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def fetch_reviews(app_key: str, start_date: datetime, end_date: datetime,
                  lang: str = "ko", country: str = "kr",
                  max_count: int = 500, progress_callback=None) -> pd.DataFrame:
    """
    start_date ~ end_date 범위의 리뷰를 수집한다.

    Google Play는 날짜 필터 API가 없어 최신순으로 페이지를 넘기며
    start_date보다 오래된 리뷰가 나올 때까지 계속 수집한다.
    max_count는 '반환할 리뷰 수' 상한이 아니라 안전장치(HARD_LIMIT 우선).
    """
    app_id = APP_IDS[app_key]
    all_reviews = []
    continuation_token = None

    start_naive = _to_naive_datetime(start_date)
    end_naive   = _to_naive_datetime(end_date)

    while len(all_reviews) < HARD_LIMIT:
        try:
            batch, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token,
            )
        except Exception as e:
            print(f"[Google Play] {app_key} 오류: {e}")
            break

        if not batch:
            break

        reached_before_start = False
        for r in batch:
            review_dt = _to_naive_datetime(r.get("at"))
            if review_dt is None:
                continue

            if review_dt < start_naive:
                reached_before_start = True
                break

            if review_dt <= end_naive:
                all_reviews.append({
                    "date":      review_dt,
                    "rating":    r.get("score", 0),
                    "title":     r.get("title", ""),
                    "text":      r.get("content", ""),
                    "user":      r.get("userName", ""),
                    "thumbs_up": r.get("thumbsUpCount", 0),
                    "app":       app_key,
                    "app_name":  APP_NAMES[app_key],
                    "store":     "Google Play",
                    "version":   r.get("reviewCreatedVersion", ""),
                })

        if progress_callback:
            progress_callback(len(all_reviews))

        if reached_before_start or not continuation_token:
            break

        time.sleep(0.3)

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
