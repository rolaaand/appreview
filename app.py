import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import os

# ── 페이지 설정 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="앱 리뷰 분석 대시보드",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 앱 메타 정보 ─────────────────────────────────────────────────────────────
APPS = {
    "melon":        {"name": "멜론",         "color": "#00CD3C", "icon": "🟢"},
    "spotify":      {"name": "Spotify",      "color": "#1A3FA3", "icon": "🎵"},
    "youtube_music":{"name": "YouTube Music","color": "#FF0000", "icon": "▶️"},
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── 캐시 유틸 ────────────────────────────────────────────────────────────────
def _cache_key(app_key, store, start, end, max_count):
    raw = f"{app_key}-{store}-{start}-{end}-{max_count}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def _cache_path(key):
    return CACHE_DIR / f"{key}.parquet"

def _load_cache(key):
    p = _cache_path(key)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            p.unlink(missing_ok=True)
    return None

def _save_cache(key, df):
    if not df.empty:
        df.to_parquet(_cache_path(key), index=False)

# ── 데이터 수집 ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(selected_apps, selected_stores, start_date, end_date):
    from collectors.google_play import fetch_reviews as gp_fetch
    from collectors.app_store  import fetch_reviews as as_fetch

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())
    frames   = []

    for app_key in selected_apps:
        if "Google Play" in selected_stores:
            key = _cache_key(app_key, "gp", start_date, end_date, 0)
            df  = _load_cache(key)
            if df is None:
                df = gp_fetch(app_key, start_dt, end_dt)
                _save_cache(key, df)
            frames.append(df)

        if "App Store" in selected_stores:
            key = _cache_key(app_key, "as", start_date, end_date, 0)
            df  = _load_cache(key)
            if df is None:
                df = as_fetch(app_key, start_dt, end_dt)
                _save_cache(key, df)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if combined.empty:
        return combined

    combined["date"] = pd.to_datetime(combined["date"])
    return combined

# ── 감성 분석 적용 ───────────────────────────────────────────────────────────
def apply_sentiment(df):
    if df.empty:
        return df
    from analyzers.sentiment import classify_sentiment
    return classify_sentiment(df)

# ── 워드클라우드 생성 ────────────────────────────────────────────────────────
def make_wordcloud(freq_dict, title=""):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # macOS 한글 폰트 탐색
        font_candidates = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]
        font_path = next((p for p in font_candidates if os.path.exists(p)), None)

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            font_path=font_path,
            max_words=80,
            colormap="viridis",
        ).generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=14)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 앱 리뷰 분석")
    st.markdown("---")

    st.subheader("📅 기간 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("종료일", value=datetime.now().date())

    if start_date > end_date:
        st.error("시작일이 종료일보다 늦을 수 없습니다.")
        st.stop()

    st.subheader("📱 앱 선택")
    selected_apps = []
    for app_key, meta in APPS.items():
        if st.checkbox(f"{meta['icon']} {meta['name']}", value=True, key=f"app_{app_key}"):
            selected_apps.append(app_key)

    st.subheader("🏪 스토어 선택")
    selected_stores = []
    if st.checkbox("Google Play", value=True):
        selected_stores.append("Google Play")
    if st.checkbox("App Store", value=True):
        selected_stores.append("App Store")

    st.subheader("⚙️ 수집 설정")
    st.caption("Google Play는 날짜 기반으로 자동 수집합니다.")
    fetch_btn = st.button("🔍 리뷰 수집 시작", use_container_width=True, type="primary")

    if st.button("🗑️ 캐시 초기화", use_container_width=True):
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        st.cache_data.clear()
        st.success("캐시가 초기화되었습니다.")
        st.rerun()

    st.markdown("---")
    st.info(
        "**App Store 수집 한계**\n\n"
        "Apple RSS API는 최근 **499건**(약 10페이지)만 제공합니다. "
        "Spotify 기준 **약 2개월 이전** 데이터는 App Store에서 수집이 불가능합니다. "
        "이는 Apple의 API 정책으로 인한 구조적 한계입니다.",
        icon="ℹ️",
    )
    st.caption("데이터 출처: Google Play / Apple App Store")

# ── 메인 영역 ────────────────────────────────────────────────────────────────
st.title("🎵 음악 앱 리뷰 분석 대시보드")
st.caption(f"분석 기간: {start_date} ~ {end_date}  |  앱: {', '.join([APPS[a]['name'] for a in selected_apps])}  |  스토어: {', '.join(selected_stores)}")

if not selected_apps:
    st.warning("좌측 사이드바에서 분석할 앱을 하나 이상 선택해 주세요.")
    st.stop()

if not selected_stores:
    st.warning("좌측 사이드바에서 스토어를 하나 이상 선택해 주세요.")
    st.stop()

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

if fetch_btn or st.session_state["df"].empty:
    with st.spinner("리뷰를 수집하는 중입니다... (첫 수집은 시간이 걸릴 수 있습니다)"):
        raw_df = fetch_all(
            tuple(selected_apps), tuple(selected_stores),
            start_date, end_date,
        )
        if not raw_df.empty:
            raw_df = apply_sentiment(raw_df)
        st.session_state["df"] = raw_df

df = st.session_state["df"]

if df.empty:
    st.info("수집된 리뷰가 없습니다. 기간이나 앱 선택을 조정해 보세요.")
    st.stop()

# 선택된 앱/스토어 필터 적용
df = df[df["app"].isin(selected_apps) & df["store"].isin(selected_stores)]

# 앱 컬러맵
COLOR_MAP = {APPS[k]["name"]: APPS[k]["color"] for k in APPS}

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 개요",
    "📈 평점 트렌드",
    "😊 감성 분석",
    "🔤 키워드 분석",
    "⚡ 앱 비교",
    "📋 원본 데이터",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: 개요
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("핵심 지표")

    kpi_cols = st.columns(len(selected_apps) * 2 if len(selected_apps) <= 3 else 6)
    col_idx = 0
    for app_key in selected_apps:
        app_df = df[df["app"] == app_key]
        app_name = APPS[app_key]["name"]
        avg_rating = app_df["rating"].mean() if not app_df.empty else 0
        total = len(app_df)
        with kpi_cols[col_idx]:
            st.metric(f"{APPS[app_key]['icon']} {app_name}", f"{total:,}개")
        with kpi_cols[col_idx + 1]:
            st.metric("평균 별점", f"⭐ {avg_rating:.2f}")
        col_idx += 2

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        # 별점 분포 (앱별)
        rating_dist = (
            df.groupby(["app_name", "rating"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            rating_dist, x="rating", y="count", color="app_name",
            barmode="group",
            title="별점 분포 비교",
            labels={"rating": "별점", "count": "리뷰 수", "app_name": "앱"},
            color_discrete_map=COLOR_MAP,
        )
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 스토어별 리뷰 수
        store_dist = df.groupby(["app_name", "store"]).size().reset_index(name="count")
        fig2 = px.bar(
            store_dist, x="app_name", y="count", color="store",
            barmode="group",
            title="앱/스토어별 리뷰 수",
            labels={"app_name": "앱", "count": "리뷰 수", "store": "스토어"},
            color_discrete_map={"Google Play": "#34A853", "App Store": "#0D96F6"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 일별 리뷰 수
    df["date_only"] = df["date"].dt.date
    daily = df.groupby(["date_only", "app_name"]).size().reset_index(name="count")
    fig3 = px.line(
        daily, x="date_only", y="count", color="app_name",
        title="일별 리뷰 수 추이",
        labels={"date_only": "날짜", "count": "리뷰 수", "app_name": "앱"},
        color_discrete_map=COLOR_MAP,
        markers=True,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: 평점 트렌드
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("평점 트렌드 분석")

    freq_opt = st.radio("집계 단위", ["일별", "주별", "월별"], horizontal=True, index=1)
    freq_map = {"일별": "D", "주별": "W", "월별": "M"}
    freq_code = freq_map[freq_opt]

    df_trend = df.copy()
    df_trend["period"] = df_trend["date"].dt.to_period(freq_code).dt.start_time

    avg_trend = (
        df_trend.groupby(["period", "app_name"])["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "review_count"})
    )

    fig = px.line(
        avg_trend, x="period", y="avg_rating", color="app_name",
        title=f"평균 별점 추이 ({freq_opt})",
        labels={"period": "기간", "avg_rating": "평균 별점", "app_name": "앱"},
        color_discrete_map=COLOR_MAP,
        markers=True,
        range_y=[1, 5],
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 별점 히트맵 (앱별)
    for app_key in selected_apps:
        app_name = APPS[app_key]["name"]
        app_df = df_trend[df_trend["app"] == app_key].copy()
        if app_df.empty:
            continue
        heatmap_data = (
            app_df.groupby(["period", "rating"])
            .size()
            .reset_index(name="count")
        )
        pivot = heatmap_data.pivot(index="rating", columns="period", values="count").fillna(0)
        fig_h = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c.date()) for c in pivot.columns],
            y=[f"⭐{int(r)}" for r in pivot.index],
            colorscale="Blues",
            text=pivot.values.astype(int),
            texttemplate="%{text}",
        ))
        fig_h.update_layout(
            title=f"{app_name} 별점 히트맵",
            xaxis_title="기간",
            yaxis_title="별점",
            height=250,
        )
        st.plotly_chart(fig_h, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: 감성 분석
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("감성 분석")

    if "sentiment" not in df.columns:
        st.warning("감성 분석 결과가 없습니다.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sent_dist = df.groupby(["app_name", "sentiment"]).size().reset_index(name="count")
            sent_pct = sent_dist.copy()
            totals = sent_pct.groupby("app_name")["count"].transform("sum")
            sent_pct["pct"] = (sent_pct["count"] / totals * 100).round(1)

            fig = px.bar(
                sent_pct, x="app_name", y="pct", color="sentiment",
                title="앱별 감성 비율 (%)",
                labels={"app_name": "앱", "pct": "비율 (%)", "sentiment": "감성"},
                color_discrete_map={"긍정": "#2ECC71", "중립": "#95A5A6", "부정": "#E74C3C"},
                text="pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            for app_key in selected_apps:
                app_name = APPS[app_key]["name"]
                app_df = df[df["app"] == app_key]
                if app_df.empty:
                    continue
                counts = app_df["sentiment"].value_counts()
                total = len(app_df)
                pos = counts.get("긍정", 0)
                neg = counts.get("부정", 0)
                neu = counts.get("중립", 0)
                fig_pie = px.pie(
                    values=[pos, neu, neg],
                    names=["긍정", "중립", "부정"],
                    title=f"{app_name} 감성 분포",
                    color_discrete_sequence=["#2ECC71", "#95A5A6", "#E74C3C"],
                    hole=0.4,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # 감성 트렌드
        st.markdown("---")
        freq_s = st.radio("집계 단위", ["일별", "주별", "월별"], horizontal=True, key="sent_freq", index=1)
        freq_code_s = {"일별": "D", "주별": "W", "월별": "M"}[freq_s]

        df_s = df.copy()
        df_s["period"] = df_s["date"].dt.to_period(freq_code_s).dt.start_time
        trend_s = (
            df_s.groupby(["period", "app_name", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        for sentiment_val, color in [("긍정", "#2ECC71"), ("부정", "#E74C3C")]:
            sub = trend_s[trend_s["sentiment"] == sentiment_val]
            fig_st = px.line(
                sub, x="period", y="count", color="app_name",
                title=f"{sentiment_val} 리뷰 추이 ({freq_s})",
                labels={"period": "기간", "count": "리뷰 수", "app_name": "앱"},
                color_discrete_map=COLOR_MAP,
                markers=True,
            )
            st.plotly_chart(fig_st, use_container_width=True)

        # ── 키워드 드릴다운 ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 키워드 드릴다운")
        st.caption("상위 키워드를 선택하면 해당 키워드가 포함된 리뷰를 서브 키워드로 재분류합니다.")

        from analyzers.keywords import extract_keywords, get_subgroup_analysis

        dd_col1, dd_col2, dd_col3 = st.columns([2, 2, 1])
        with dd_col1:
            dd_app = st.selectbox(
                "앱 선택",
                [APPS[k]["name"] for k in selected_apps],
                key="dd_app",
            )
        dd_app_key = next(k for k, v in APPS.items() if v["name"] == dd_app)
        dd_df = df[df["app"] == dd_app_key]

        with dd_col2:
            dd_sent_filter = st.radio(
                "감성 필터",
                ["전체", "긍정", "부정", "중립"],
                horizontal=True,
                key="dd_sent",
            )

        top_kws = extract_keywords(
            dd_df,
            top_n=20,
            sentiment_filter=None if dd_sent_filter == "전체" else dd_sent_filter,
        )

        with dd_col3:
            dd_top_n = st.slider("키워드 수", 5, 20, 10, key="dd_top_n")

        if not top_kws:
            st.info("키워드를 추출할 리뷰가 없습니다.")
        else:
            # 상위 키워드 버튼 선택
            kw_labels = [f"{kw} ({cnt})" for kw, cnt in top_kws[:dd_top_n]]
            selected_kw_label = st.radio(
                "📌 키워드 선택",
                kw_labels,
                horizontal=True,
                key="dd_kw_select",
            )
            selected_kw = selected_kw_label.split(" (")[0]

            result = get_subgroup_analysis(dd_df, selected_kw, top_n=15)
            matched_df  = result["matched_df"]
            subkeywords = result["subkeywords"]
            subgroups   = result["subgroups"]

            if matched_df.empty:
                st.info(f"'{selected_kw}' 키워드가 포함된 리뷰가 없습니다.")
            else:
                # KPI 요약
                total_matched = len(matched_df)
                sent_counts = matched_df["sentiment"].value_counts() if "sentiment" in matched_df.columns else {}
                pos_n = int(sent_counts.get("긍정", 0))
                neg_n = int(sent_counts.get("부정", 0))
                neu_n = int(sent_counts.get("중립", 0))
                avg_r = matched_df["rating"].mean()

                mk1, mk2, mk3, mk4, mk5 = st.columns(5)
                mk1.metric("매칭 리뷰", f"{total_matched:,}개")
                mk2.metric("평균 별점", f"⭐ {avg_r:.2f}")
                mk3.metric("😊 긍정", f"{pos_n}개  ({pos_n/total_matched*100:.0f}%)")
                mk4.metric("😐 중립", f"{neu_n}개  ({neu_n/total_matched*100:.0f}%)")
                mk5.metric("😠 부정", f"{neg_n}개  ({neg_n/total_matched*100:.0f}%)")

                st.markdown(f"#### `{selected_kw}` 연관 서브 키워드")

                sc1, sc2 = st.columns([1, 1])
                with sc1:
                    # 서브 키워드 가로 막대
                    if subkeywords:
                        sub_df_plot = pd.DataFrame(subkeywords[:15], columns=["서브키워드", "언급수"])
                        fig_sub = px.bar(
                            sub_df_plot, x="언급수", y="서브키워드",
                            orientation="h",
                            title="연관 서브 키워드 빈도",
                            color="언급수",
                            color_continuous_scale="Blues",
                        )
                        fig_sub.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            height=420,
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig_sub, use_container_width=True)

                with sc2:
                    # 서브 키워드별 감성 분포 (스택 바)
                    if subgroups:
                        sg_rows = []
                        for sg in subgroups:
                            for sent_label, cnt in sg["sentiment_dist"].items():
                                sg_rows.append({
                                    "서브키워드": sg["keyword"],
                                    "감성": sent_label,
                                    "count": cnt,
                                })
                        sg_df = pd.DataFrame(sg_rows)
                        if not sg_df.empty:
                            fig_sg = px.bar(
                                sg_df, x="count", y="서브키워드", color="감성",
                                orientation="h",
                                title="서브 키워드별 감성 분포",
                                color_discrete_map={
                                    "긍정": "#2ECC71",
                                    "중립": "#95A5A6",
                                    "부정": "#E74C3C",
                                },
                                barmode="stack",
                            )
                            fig_sg.update_layout(
                                yaxis={"categoryorder": "total ascending"},
                                height=420,
                            )
                            st.plotly_chart(fig_sg, use_container_width=True)

                # 서브 키워드별 실제 댓글 (expander)
                st.markdown("#### 📝 서브 키워드별 댓글 상세")
                st.caption("각 리뷰는 가장 관련도 높은 키워드 하나에만 배치됩니다. 공감순(👍) → 내용 길이순으로 정렬됩니다.")
                SENT_ICON  = {"긍정": "😊", "중립": "😐", "부정": "😠", "": ""}
                SENT_COLOR = {"긍정": "#d4edda", "중립": "#f8f9fa", "부정": "#f8d7da", "": "#f8f9fa"}

                for sg in subgroups:
                    owned = sg.get("owned_count", len(sg["samples"]))
                    header = (
                        f"**{sg['keyword']}** — 언급 {sg['count']}건 / 전용 댓글 {owned}건  |  "
                        f"😊 {sg['sentiment_dist']['긍정']}  "
                        f"😐 {sg['sentiment_dist']['중립']}  "
                        f"😠 {sg['sentiment_dist']['부정']}"
                    )
                    with st.expander(header, expanded=False):
                        if not sg["samples"]:
                            st.caption("이 키워드에 단독으로 배정된 리뷰가 없습니다.")
                            continue
                        for sample in sg["samples"]:
                            icon  = SENT_ICON.get(sample["sentiment"], "")
                            stars = "⭐" * sample["rating"]
                            bg    = SENT_COLOR.get(sample["sentiment"], "#f8f9fa")
                            tu    = sample.get("thumbs_up", 0)
                            tu_badge = f' &nbsp;<small>👍 {tu}</small>' if tu > 0 else ""
                            st.markdown(
                                f"<div style='background:{bg};color:#212529;"
                                f"border-radius:6px;padding:8px 12px;margin-bottom:6px;'>"
                                f"{icon} {stars}{tu_badge} &nbsp; "
                                f"<small><b>{sample['app_name']}</b> · {sample['store']} · {sample['date']}</small><br/>"
                                f"{sample['text']}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: 키워드 분석
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("키워드 분석")

    from analyzers.keywords import extract_keywords, get_keyword_trend

    kw_app   = st.selectbox("앱 선택", [APPS[k]["name"] for k in selected_apps], key="kw_app")
    kw_sent  = st.radio("감성 필터", ["전체", "긍정", "부정", "중립"], horizontal=True, key="kw_sent")
    kw_n     = st.slider("키워드 수", 10, 50, 30, key="kw_n")

    kw_app_key = next(k for k, v in APPS.items() if v["name"] == kw_app)
    kw_df = df[df["app"] == kw_app_key]

    sent_filter = None if kw_sent == "전체" else kw_sent
    keywords = extract_keywords(kw_df, top_n=kw_n, sentiment_filter=sent_filter)

    if keywords:
        freq_dict = dict(keywords)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_wc = make_wordcloud(freq_dict, title=f"{kw_app} 키워드 워드클라우드")
            if fig_wc:
                st.pyplot(fig_wc)
            else:
                st.info("워드클라우드를 생성할 수 없습니다 (wordcloud 패키지 필요).")

        with c2:
            kw_df_plot = pd.DataFrame(keywords, columns=["keyword", "count"])
            fig_bar = px.bar(
                kw_df_plot, x="count", y="keyword",
                orientation="h",
                title="상위 키워드",
                labels={"count": "언급 횟수", "keyword": "키워드"},
                color="count",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 키워드 트렌드 (상위 5개)
        if len(keywords) >= 3:
            st.markdown("---")
            top_kws = [k for k, _ in keywords[:5]]
            kw_trend = get_keyword_trend(kw_df, top_kws, freq="W")
            if not kw_trend.empty:
                fig_kt = px.line(
                    kw_trend, x="period", y="count", color="keyword",
                    title=f"{kw_app} 상위 키워드 주간 트렌드",
                    labels={"period": "주", "count": "언급 수", "keyword": "키워드"},
                    markers=True,
                )
                st.plotly_chart(fig_kt, use_container_width=True)
    else:
        st.info("키워드를 추출할 수 있는 리뷰가 없습니다.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5: 앱 비교
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("앱 간 비교 분석")

    if len(selected_apps) < 2:
        st.info("비교 분석을 위해 2개 이상의 앱을 선택해 주세요.")
    else:
        comparison_rows = []
        for app_key in selected_apps:
            app_df = df[df["app"] == app_key]
            if app_df.empty:
                continue

            pos = len(app_df[app_df.get("sentiment", pd.Series()) == "긍정"]) if "sentiment" in app_df.columns else 0
            neg = len(app_df[app_df.get("sentiment", pd.Series()) == "부정"]) if "sentiment" in app_df.columns else 0

            comparison_rows.append({
                "앱": APPS[app_key]["name"],
                "총 리뷰": len(app_df),
                "평균 별점": round(app_df["rating"].mean(), 2),
                "5점 비율(%)": round((app_df["rating"] == 5).mean() * 100, 1),
                "1점 비율(%)": round((app_df["rating"] == 1).mean() * 100, 1),
                "긍정 비율(%)": round(pos / len(app_df) * 100, 1) if len(app_df) else 0,
                "부정 비율(%)": round(neg / len(app_df) * 100, 1) if len(app_df) else 0,
            })

        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df.set_index("앱"), use_container_width=True)

        st.markdown("---")

        # 레이더 차트
        metrics = ["평균 별점", "5점 비율(%)", "긍정 비율(%)"]
        fig_radar = go.Figure()
        for _, row in comp_df.iterrows():
            app_name = row["앱"]
            vals = [row[m] for m in metrics]
            # normalize: 평균 별점 /5 *100, 나머지는 그대로
            norm_vals = [row["평균 별점"] / 5 * 100, row["5점 비율(%)"], row["긍정 비율(%)"] ]
            norm_vals.append(norm_vals[0])
            cats = metrics + [metrics[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals, theta=cats,
                fill="toself", name=app_name,
                line_color=COLOR_MAP.get(app_name),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="앱 비교 레이더 차트",
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 앱별 상위 키워드 비교
        from analyzers.keywords import extract_keywords
        st.markdown("---")
        st.subheader("앱별 상위 키워드 비교")
        kw_cols = st.columns(len(selected_apps))
        for i, app_key in enumerate(selected_apps):
            app_name = APPS[app_key]["name"]
            app_df = df[df["app"] == app_key]
            kws = extract_keywords(app_df, top_n=15)
            with kw_cols[i]:
                st.markdown(f"**{app_name}**")
                if kws:
                    kw_table = pd.DataFrame(kws, columns=["키워드", "횟수"])
                    st.dataframe(kw_table, use_container_width=True, height=400)
                else:
                    st.info("키워드 없음")

# ════════════════════════════════════════════════════════════════════════════
# TAB 6: 원본 데이터
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("원본 데이터")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_app = st.multiselect("앱 필터", [APPS[k]["name"] for k in selected_apps],
                                    default=[APPS[k]["name"] for k in selected_apps])
    with fc2:
        filter_store = st.multiselect("스토어 필터", selected_stores, default=selected_stores)
    with fc3:
        filter_rating = st.multiselect("별점 필터", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    display_df = df[
        df["app_name"].isin(filter_app) &
        df["store"].isin(filter_store) &
        df["rating"].isin(filter_rating)
    ].copy()

    search_kw = st.text_input("🔍 키워드 검색 (리뷰 내용)")
    if search_kw:
        display_df = display_df[display_df["text"].str.contains(search_kw, case=False, na=False)]

    st.caption(f"총 {len(display_df):,}개 리뷰")

    show_cols = ["date", "app_name", "store", "rating", "sentiment", "title", "text", "user"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    st.dataframe(
        display_df[show_cols].sort_values("date", ascending=False),
        use_container_width=True,
        height=500,
    )

    # CSV 다운로드
    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 CSV 다운로드",
        data=csv,
        file_name=f"app_reviews_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
