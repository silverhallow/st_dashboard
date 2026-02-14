import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# 페이지 설정
st.set_page_config(page_title="네이버 쇼핑 데이터 대시보드", layout="wide")

# 한국어 폰트 설정 (Plotly는 시스템 폰트를 따르지만 레이블 확인 필요)
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 1. 데이터 로드 함수
@st.cache_data
def load_data():
    today_str = "20260213"
    # GitHub 배포 경로 대응 (naverapieda/data) 또는 로컬 경로 (data)
    data_dir = "naverapieda/data" if os.path.exists("naverapieda/data") else "data"
    
    keywords = ["오메가3", "비타민d"]
    files = {
        "trend": {kw: f"{kw}_트렌드_수집일자_{today_str}.csv" for kw in keywords},
        "blog": {kw: f"{kw}_블로그_수집일자_{today_str}.csv" for kw in keywords},
        "shop": {kw: f"{kw}_네이버쇼핑_수집일자_{today_str}.csv" for kw in keywords},
    }
    
    data = {"trend": {}, "blog": {}, "shop": {}}
    for category, kw_dict in files.items():
        for kw, filename in kw_dict.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if category == "trend":
                    df['period'] = pd.to_datetime(df['period'])
                if category == "shop":
                    df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
                data[category][kw] = df
    return data

data = load_data()

# 사이드바
st.sidebar.title("🔍 검색 및 필터")
selected_keywords = st.sidebar.multiselect(
    "분석할 키워드를 선택하세요",
    options=["오메가3", "비타민d"],
    default=["오메가3", "비타민d"]
)

st.sidebar.divider()
st.sidebar.info("네이버 API로 수집된 최근 1년 데이터를 기반으로 합니다.")

# 메인 타이틀
st.title("📊 네이버 쇼핑 데이터 분석 대시보드")
st.write("오메가3와 비타민D에 대한 쇼핑 트렌드, 검색 결과 및 블로그 동향을 한눈에 파악할 수 있습니다.")

if not selected_keywords:
    st.warning("키워드를 선택해주세요.")
else:
    tab1, tab2, tab3 = st.tabs(["📉 트렌드 비교", "🛍️ 쇼핑 데이터 분석", "📝 블로그 데이터 분석"])

    # --- Tab 1: 트렌드 비교 ---
    with tab1:
        st.header("쇼핑 클릭 트렌드 비교")
        
        # 1. 메트릭 표시 (상단)
        cols = st.columns(len(selected_keywords))
        for i, kw in enumerate(selected_keywords):
            df = data['trend'][kw]
            latest_ratio = df.iloc[-1]['ratio']
            prev_ratio = df.iloc[-2]['ratio']
            delta = round(latest_ratio - prev_ratio, 2)
            cols[i].metric(label=f"{kw} 현재 클릭 지수", value=f"{latest_ratio:.2f}", delta=f"{delta}")

        # 2. 트렌드 그래프 (Plotly Line)
        fig_trend = go.Figure()
        for kw in selected_keywords:
            df = data['trend'][kw]
            fig_trend.add_trace(go.Scatter(x=df['period'], y=df['ratio'], name=kw, mode='lines'))
        
        fig_trend.update_layout(
            title="최근 1년 키워드별 쇼핑 클릭 지수 추이",
            xaxis_title="날짜",
            yaxis_title="상대적 클릭수 (최대 100)",
            legend_title="키워드",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.write("> **해석**: 선택된 키워드 간의 클릭 추이를 확인할 수 있습니다. 주기적인 피크 현상은 주간 또는 월간 소비 패턴을 반영합니다.")

        # 3. 기술 통계표
        st.subheader("키워드별 트렌드 요약 통계")
        trend_stats = pd.concat([data['trend'][kw]['ratio'].describe().rename(kw) for kw in selected_keywords], axis=1)
        st.dataframe(trend_stats.T, use_container_width=True)
        
        # 4. 실시간 주요 지표 (Max, Min) - Table
        st.subheader("임계치 데이터 (최대/최소 클릭 일자)")
        peak_list = []
        for kw in selected_keywords:
            df = data['trend'][kw]
            max_row = df.loc[df['ratio'].idxmax()]
            min_row = df.loc[df['ratio'].idxmin()]
            peak_list.append({"키워드": kw, "최대값": max_row['ratio'], "최대일자": max_row['period'].date(), "최소값": min_row['ratio'], "최소일자": min_row['period'].date()})
        st.table(pd.DataFrame(peak_list))
        
        # 5. 상관관계 분석 (다변량) - Table
        if len(selected_keywords) > 1:
            st.subheader("키워드 간 상관관계")
            corr_df = pd.concat([data['trend'][kw].set_index('period')['ratio'].rename(kw) for kw in selected_keywords], axis=1).corr()
            st.dataframe(corr_df)
            st.write("지수가 1에 가까울수록 두 키워드의 클릭 추이가 동일하게 움직임을 의미합니다.")

    # --- Tab 2: 쇼핑 데이터 분석 ---
    with tab2:
        st.header("네이버 쇼핑 상품 데이터 분석")
        
        kw_col = st.selectbox("집중 분석 키워드 선택", options=selected_keywords)
        df_shop = data['shop'][kw_col]
        
        c1, c2 = st.columns(2)
        
        with c1:
            # 1. 가격 분포 (Histogram)
            fig_price = px.histogram(df_shop, x="lprice", nbins=20, title=f"[{kw_col}] 상품 최저가 분포", labels={"lprice": "최저가", "count": "상품 수"})
            st.plotly_chart(fig_price, use_container_width=True)
            st.write("주요 가격대를 파악하여 시장의 보급형과 프리미엄 제품군 비중을 확인할 수 있습니다.")
            
            # 2. 브랜드 점유율 (Pie)
            brand_counts = df_shop['brand'].fillna("미지정").value_counts().head(10)
            fig_brand = px.pie(values=brand_counts.values, names=brand_counts.index, title=f"[{kw_col}] 상위 10개 브랜드 노출 비중")
            st.plotly_chart(fig_brand, use_container_width=True)

        with c2:
            # 3. 쇼핑몰 빈도 (Bar)
            mall_counts = df_shop['mallName'].value_counts().head(10)
            fig_mall = px.bar(x=mall_counts.index, y=mall_counts.values, title=f"[{kw_col}] 상위 노출 쇼핑몰 빈도", labels={"x": "쇼핑몰", "y": "상품 수"})
            st.plotly_chart(fig_mall, use_container_width=True)
            st.write("어떤 유통 채널에서 해당 키워드 상품이 가장 활발하게 경쟁 중인지 확인 가능합니다.")
            
            # 4. 쇼핑몰별 평균 가격 (Table)
            st.subheader("쇼핑몰별 평균 판매가 요약")
            mall_avg_price = df_shop.groupby('mallName')['lprice'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
            st.dataframe(mall_avg_price, use_container_width=True)

        # 5. 상품 리스트 (Table)
        st.subheader("검색 결과 상위 상품 리스트")
        st.dataframe(df_shop[['title', 'mallName', 'lprice', 'brand', 'link']], use_container_width=True)

    # --- Tab 3: 블로그 데이터 분석 ---
    with tab3:
        st.header("블로그 동향 및 키워드 분석")
        
        kw_blog = st.selectbox("블로그 데이터 키워드 선택", options=selected_keywords, key="blog_select")
        df_blog = data['blog'][kw_blog]
        
        # 1. TF-IDF 키워드 추출
        vectorizer = TfidfVectorizer(max_features=20)
        tfidf = vectorizer.fit_transform(df_blog['title'].fillna(''))
        keywords_list = vectorizer.get_feature_names_out()
        weights = tfidf.toarray().sum(axis=0)
        keyword_df = pd.DataFrame({'keyword': keywords_list, 'weight': weights}).sort_values('weight', ascending=False)
        
        # 2. 키워드 분석 그래프 (Horizontal Bar)
        fig_kw = px.bar(keyword_df, y='keyword', x='weight', orientation='h', title=f"[{kw_blog}] 블로그 제목 핵심 키워드 (TF-IDF)", labels={"weight": "가중치", "keyword": "키워드"})
        st.plotly_chart(fig_kw, use_container_width=True)
        st.write("블로그 포스팅 제목에서 공통적으로 발견되는 관심사를 정량적으로 분석한 결과입니다.")
        
        # 3. 포스트 리스트 (Table)
        st.subheader("최신/정합도순 블로그 포스트 리스트")
        st.dataframe(df_blog[['title', 'bloggername', 'postdate', 'link']], use_container_width=True)
        
        # 4. 블로거별 포스팅 빈도 (Table)
        st.subheader("주요 블로거 활동 현황")
        blogger_counts = df_blog['bloggername'].value_counts().head(10).to_frame(name="포스팅 수")
        st.table(blogger_counts)
        
        # 5. 정보성 키워드 매칭 통계 (Table)
        st.subheader("주제별 키워드 매칭 가중치")
        st.dataframe(keyword_df.head(10).reset_index(drop=True))
        st.write("정보 전달 수준이 높은 블로그 포스트의 주제 구성을 유추할 수 있습니다.")
