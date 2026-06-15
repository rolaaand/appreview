import re
from collections import Counter
from typing import List, Tuple
import pandas as pd

# 불용어 (한국어 + 영어)
STOPWORDS_KO = {
    "이", "가", "을", "를", "은", "는", "의", "에", "서", "로", "으로",
    "와", "과", "도", "만", "에서", "에게", "한테", "하고", "이고", "이며",
    "그", "이", "저", "것", "수", "하다", "있다", "없다", "되다", "하여",
    "해서", "그래서", "그리고", "하지만", "근데", "그냥", "좀", "더", "잘",
    "제", "저", "너무", "진짜", "정말", "아주", "매우", "다", "안", "못",
    "앱", "어플", "어플리케이션", "사용", "이용", "기능", "버전", "업데이트",
    "음악", "노래", "멜론", "스포티파이", "유튜브", "뮤직",
    "하는", "있는", "없는", "되는", "같은", "다른", "많은",
    "그냥", "근데", "좀", "뭔", "뭐", "왜", "어떻게",
    "합니다", "합니다만", "됩니다", "됩니다만", "입니다",
    "않다", "않고", "않아", "않아서", "않으면",
    "때", "때문", "경우", "시", "동안",
}

STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "this", "that", "these", "those", "it", "its", "my", "your", "our",
    "i", "you", "we", "they", "he", "she", "app", "music", "song",
    "and", "or", "but", "if", "so", "yet", "nor",
}


def _try_kiwi_tokenize(texts: List[str]) -> List[str]:
    """kiwipiepy 형태소 분석기 (설치된 경우 사용)"""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        tokens = []
        for text in texts:
            result = kiwi.analyze(text)
            for sent in result[0][0]:
                word = sent.form
                tag = sent.tag
                # 명사, 동사, 형용사만 추출
                if tag.startswith(("NN", "VV", "VA", "XR")) and len(word) > 1:
                    tokens.append(word)
        return tokens
    except ImportError:
        return []


def _simple_tokenize(texts: List[str]) -> List[str]:
    """kiwipiepy 없을 때 간단한 토크나이징"""
    tokens = []
    for text in texts:
        # 특수문자 제거 후 공백/구두점 분리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        words = text.split()
        for word in words:
            word = word.strip().lower()
            if len(word) >= 2:
                tokens.append(word)
    return tokens


def extract_keywords(df: pd.DataFrame, top_n: int = 30,
                     sentiment_filter: str = None) -> List[Tuple[str, int]]:
    """리뷰 텍스트에서 상위 키워드 추출"""
    if df.empty:
        return []

    filtered = df.copy()
    if sentiment_filter and "sentiment" in df.columns:
        filtered = df[df["sentiment"] == sentiment_filter]

    texts = filtered["text"].dropna().tolist()
    if not texts:
        return []

    # 형태소 분석 시도 (kiwipiepy)
    tokens = _try_kiwi_tokenize(texts)
    use_kiwi = len(tokens) > 0

    if not use_kiwi:
        tokens = _simple_tokenize(texts)

    # 불용어 제거
    filtered_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in STOPWORDS_KO:
            continue
        if token_lower in STOPWORDS_EN:
            continue
        if len(token) < 2:
            continue
        if re.match(r'^\d+$', token):
            continue
        filtered_tokens.append(token)

    counter = Counter(filtered_tokens)
    return counter.most_common(top_n)


def extract_keywords_by_app(df: pd.DataFrame, top_n: int = 20) -> dict:
    """앱별 키워드 추출"""
    result = {}
    for app_key in df["app"].unique():
        app_df = df[df["app"] == app_key]
        result[app_key] = extract_keywords(app_df, top_n=top_n)
    return result


def _review_priority_key(row) -> tuple:
    """리뷰 우선순위: thumbs_up 내림차순 → 텍스트 길이 내림차순"""
    return (-int(row.get("thumbs_up", 0)), -len(str(row.get("text", ""))))


def get_subgroup_analysis(df: pd.DataFrame, keyword: str,
                          top_n: int = 15) -> dict:
    """
    keyword 포함 리뷰를 서브 키워드별로 분류한다.

    핵심 규칙:
    - 각 리뷰는 가장 관련도 높은 서브 키워드 그룹 하나에만 배치 (중복 없음)
    - 관련도 = 해당 리뷰에서 서브 키워드가 등장하는 횟수가 가장 많은 그룹
      (동점이면 전체 빈도가 높은 그룹 우선)
    - 그룹 내 정렬: thumbs_up 내림차순 → 텍스트 길이 내림차순
    """
    if df.empty or not keyword:
        return {"matched_df": pd.DataFrame(), "subkeywords": [], "subgroups": []}

    keyword_lower = keyword.lower()
    matched = df[df["text"].str.lower().str.contains(keyword_lower, na=False)].copy()
    matched = matched.reset_index(drop=True)

    if matched.empty:
        return {"matched_df": matched, "subkeywords": [], "subgroups": []}

    texts = matched["text"].dropna().tolist()

    # 서브 키워드 추출 (선택 키워드 자체 제외)
    tokens = _try_kiwi_tokenize(texts) or _simple_tokenize(texts)
    filtered_tokens = []
    for token in tokens:
        tl = token.lower()
        if tl in STOPWORDS_KO or tl in STOPWORDS_EN:
            continue
        if len(token) < 2 or re.match(r'^\d+$', token):
            continue
        if tl == keyword_lower:
            continue
        filtered_tokens.append(token)

    subkeywords = Counter(filtered_tokens).most_common(top_n)
    if not subkeywords:
        return {"matched_df": matched, "subkeywords": [], "subgroups": []}

    sub_kw_list = [kw for kw, _ in subkeywords[:10]]
    # 전체 빈도 순위 맵 (낮을수록 중요)
    kw_rank = {kw: i for i, kw in enumerate(sub_kw_list)}

    # ── 각 리뷰를 하나의 서브 키워드에만 할당 ────────────────────────────
    # 리뷰별로 포함된 서브 키워드 목록 계산 → 빈도 가장 높은 것에 배치
    assignment: dict[int, str] = {}   # row_index → 담당 서브 키워드
    for idx, row in matched.iterrows():
        text_lower = str(row.get("text", "")).lower()
        # 이 리뷰에 포함된 서브 키워드와 그 등장 횟수
        hits = {}
        for kw in sub_kw_list:
            cnt = text_lower.count(kw.lower())
            if cnt > 0:
                hits[kw] = cnt
        if not hits:
            continue
        # 등장 횟수 내림차순, 동점이면 전체 빈도 순위 오름차순
        best = sorted(hits.keys(),
                      key=lambda k: (-hits[k], kw_rank[k]))[0]
        assignment[idx] = best

    # ── 서브 키워드별 그룹 조립 ──────────────────────────────────────────
    # 감성 분포는 해당 키워드를 포함한 전체 리뷰 기준 (중복 허용)
    # 샘플 리뷰는 독점 배정된 리뷰만 사용 (중복 없음)
    subgroups = []
    for sub_kw in sub_kw_list:
        sub_lower = sub_kw.lower()
        # 감성 분포: 서브 키워드를 포함한 모든 리뷰
        all_sub = matched[matched["text"].str.lower().str.contains(sub_lower, na=False)]
        if all_sub.empty:
            continue

        sent_dist = {"긍정": 0, "중립": 0, "부정": 0}
        if "sentiment" in all_sub.columns:
            for s, c in all_sub["sentiment"].value_counts().items():
                if s in sent_dist:
                    sent_dist[s] = int(c)

        # 샘플: 이 그룹에 독점 배정된 리뷰만
        owned_idx = [i for i, kw in assignment.items() if kw == sub_kw]
        owned = matched.loc[owned_idx].copy()

        # 우선순위 정렬: thumbs_up 내림차순 → 텍스트 길이 내림차순
        owned = owned.copy()
        owned["_tu"]  = owned["thumbs_up"].fillna(0).astype(int)
        owned["_tlen"] = owned["text"].fillna("").str.len()
        owned = owned.sort_values(["_tu", "_tlen"], ascending=[False, False])

        sample_rows = []
        for _, row in owned.iterrows():
            sample_rows.append({
                "rating":    int(row.get("rating", 0)),
                "sentiment": row.get("sentiment", ""),
                "app_name":  row.get("app_name", ""),
                "store":     row.get("store", ""),
                "date":      str(row.get("date", ""))[:10],
                "thumbs_up": int(row.get("thumbs_up", 0)),
                "text":      str(row.get("text", ""))[:300],
            })

        subgroups.append({
            "keyword":        sub_kw,
            "count":          len(all_sub),
            "owned_count":    len(owned),
            "sentiment_dist": sent_dist,
            "samples":        sample_rows,
        })

    return {
        "matched_df":  matched,
        "subkeywords": subkeywords,
        "subgroups":   subgroups,
    }


def get_keyword_trend(df: pd.DataFrame, keywords: List[str],
                      freq: str = "W") -> pd.DataFrame:
    """특정 키워드의 시간별 언급 빈도"""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period(freq).dt.start_time

    rows = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        period_counts = (
            df[df["text"].str.lower().str.contains(keyword_lower, na=False)]
            .groupby("period")
            .size()
            .reset_index(name="count")
        )
        period_counts["keyword"] = keyword
        rows.append(period_counts)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)
