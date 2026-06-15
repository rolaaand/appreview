import pandas as pd

# 한국어 감성 키워드 사전
POSITIVE_KO = {
    "좋아", "좋음", "최고", "완벽", "편리", "편함", "깔끔", "훌륭", "만족", "추천",
    "좋습니다", "좋네요", "편리하네요", "잘돼요", "잘됩니다", "잘되네요", "유용",
    "쉽고", "빠르고", "빠름", "빨라", "쾌적", "안정", "안정적", "부드럽",
    "개선", "향상", "업데이트", "고마워", "감사", "사랑", "행복", "즐거",
    "신청곡", "노래", "음질", "고음질", "청취", "감동", "좋은", "편한",
    "흘러나와", "끊기지", "잘나와", "좋아요", "너무좋아", "최고예요",
}

NEGATIVE_KO = {
    "나쁘", "별로", "최악", "불편", "어려", "복잡", "느리", "끊겨", "안돼",
    "오류", "에러", "버그", "오작동", "튕겨", "강제종료", "종료됨", "먹통",
    "안됩니다", "안되요", "안돼요", "실망", "짜증", "불만", "환불",
    "삭제", "탈퇴", "못쓰겠", "쓸수없", "쓰레기", "돈낭비", "사기",
    "광고", "광고가", "결제오류", "결제안", "끊김", "느려", "렉", "버벅",
    "문제", "이슈", "해결", "개선해", "고쳐", "수정해", "언제고쳐",
}

POSITIVE_EN = {
    "great", "good", "excellent", "amazing", "awesome", "love", "perfect",
    "best", "fantastic", "wonderful", "helpful", "easy", "fast", "smooth",
    "reliable", "stable", "recommend", "happy", "enjoy", "nice", "useful",
}

NEGATIVE_EN = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "crash",
    "bug", "error", "slow", "laggy", "freeze", "broken", "useless",
    "annoying", "disappointed", "frustrating", "problem", "issue", "fix",
}


def classify_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    별점 기반 감성 분류 + 텍스트 키워드 보정
    1-2점: 부정, 3점: 중립, 4-5점: 긍정
    텍스트에 강한 부정/긍정 키워드가 있으면 보정
    """
    if df.empty:
        return df

    result = df.copy()
    sentiments = []
    scores = []

    for _, row in result.iterrows():
        rating = row.get("rating", 3)
        text = str(row.get("text", "")).lower()

        # 별점 기반 기본 감성
        if rating >= 4:
            base = "긍정"
            base_score = rating / 5.0
        elif rating <= 2:
            base = "부정"
            base_score = rating / 5.0
        else:
            base = "중립"
            base_score = 0.5

        # 텍스트 키워드 보정
        pos_count = sum(1 for kw in POSITIVE_KO if kw in text)
        pos_count += sum(1 for kw in POSITIVE_EN if kw in text)
        neg_count = sum(1 for kw in NEGATIVE_KO if kw in text)
        neg_count += sum(1 for kw in NEGATIVE_EN if kw in text)

        if neg_count > pos_count + 2 and base == "긍정":
            base = "중립"
        elif pos_count > neg_count + 2 and base == "부정":
            base = "중립"

        sentiments.append(base)
        scores.append(round(base_score, 3))

    result["sentiment"] = sentiments
    result["sentiment_score"] = scores
    return result


def sentiment_summary(df: pd.DataFrame) -> dict:
    if df.empty or "sentiment" not in df.columns:
        return {"긍정": 0, "중립": 0, "부정": 0}
    counts = df["sentiment"].value_counts().to_dict()
    return {
        "긍정": counts.get("긍정", 0),
        "중립": counts.get("중립", 0),
        "부정": counts.get("부정", 0),
    }
