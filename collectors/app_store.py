"""
Apple App Store 리뷰 수집기
iTunes 공개 RSS API 사용 (외부 라이브러리 불필요)
"""
import requests
import pandas as pd
from datetime import datetime, timezone
import time

APP_IDS = {
    "melon":         "415597317",
    "spotify":       "324684580",
    "youtube_music": "1017492454",
}

APP_NAMES = {
    "melon":         "멜론",
    "spotify":       "Spotify",
    "youtube_music": "YouTube Music",
}

RSS_URL = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"


def _to_naive_datetime(dt_str: str):
    """ISO 8601 문자열을 timezone-naive datetime으로 변환"""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def fetch_reviews(app_key: str, start_date: datetime, end_date: datetime,
                  country: str = "kr", max_count: int = 500,
                  progress_callback=None) -> pd.DataFrame:
    """
    iTunes RSS API를 이용해 App Store 리뷰 수집
    최대 10페이지 × 50개 = 500개 (Apple 제한)
    """
    app_id   = APP_IDS[app_key]
    app_name = APP_NAMES[app_key]

    start_naive = start_date if not hasattr(start_date, 'tzinfo') or start_date.tzinfo is None \
        else start_date.astimezone(timezone.utc).replace(tzinfo=None)
    end_naive   = end_date if not hasattr(end_date, 'tzinfo') or end_date.tzinfo is None \
        else end_date.astimezone(timezone.utc).replace(tzinfo=None)

    all_reviews = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(1, 11):          # Apple은 최대 10페이지
        if len(all_reviews) >= max_count:
            break

        url = RSS_URL.format(country=country, page=page, app_id=app_id)
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception as e:
            print(f"[App Store] {app_key} p{page} 오류: {e}")
            break

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break

        # 첫 번째 entry는 앱 정보 (리뷰 아님)
        if page == 1 and entries:
            entries = entries[1:]

        reached_before_start = False
        for entry in entries:
            updated_str = entry.get("updated", {}).get("label", "")
            review_dt   = _to_naive_datetime(updated_str)

            if review_dt is None:
                continue
            if review_dt < start_naive:
                reached_before_start = True
                break
            if review_dt > end_naive:
                continue

            rating_str = entry.get("im:rating", {}).get("label", "3")
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 3

            title   = entry.get("title",   {}).get("label", "")
            content = entry.get("content", {}).get("label", "")
            author  = entry.get("author",  {}).get("name",  {}).get("label", "")
            version = entry.get("im:version", {}).get("label", "")

            all_reviews.append({
                "date":     review_dt,
                "rating":   rating,
                "title":    title,
                "text":     content,
                "user":     author,
                "thumbs_up": 0,
                "app":      app_key,
                "app_name": app_name,
                "store":    "App Store",
                "version":  version,
            })

            if progress_callback:
                progress_callback(len(all_reviews))

        if reached_before_start:
            break

        time.sleep(0.5)

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
