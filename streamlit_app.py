import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import glob
import os
from streamlit_plotly_events import plotly_events

# ====================================================================================
st.set_page_config(
    page_title="감성분석으로 본 미디어의 편향성",
    layout="wide"
)
st.markdown("""
<style>
    .block-container { max-width: 1000px; }
    h1 { font-size: 2.5em; text-align: center; font-weight: bold; }
    h2 { font-size: 2em; text-align: center; margin-top: 3em; border-bottom: 2px solid #eee; padding-bottom: 0.5em;}
    h3 { font-size: 1.5em; text-align: left; margin-top: 2em; }
    p { font-size: 1.15em !important; line-height: 1.7 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        sentiment_df = pd.read_csv('sentiment.csv')
        sentiment_df.rename(columns={'주제': 'topic', '언론사': 'outlet', '제목': 'title', '감정점수': 'sentiment', 'zscore': 'zscore', 'URL': 'url'}, inplace=True)
        
        freq_files = glob.glob(os.path.join('frequency_csvs', '*.csv'))
        freq_data = {}
        for file_path in freq_files:
            filename = os.path.basename(file_path).replace('.csv', '')
            parts = filename.split('_')
            if len(parts) == 2:
                topic, outlet = parts[0], parts[1]
                df = pd.read_csv(file_path)
                freq_data.setdefault(topic, {})[outlet] = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        
        pca_df = pd.read_csv('pca_results.csv', index_col='outlet')
        
        return sentiment_df, freq_data, pca_df
    except FileNotFoundError as e:
        st.error(f"데이터 파일 로딩 오류: {e.filename} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None, None, None

def load_markdown(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"오류: {filepath} 파일을 찾을 수 없습니다. 파일이 content 폴더 안에 있는지 확인해주세요."

def display_wordcloud(topic, freq_data):
    st.subheader(f"주요 키워드 분석: '{topic}'")
    if freq_data and topic in freq_data:
        available_outlets = sorted(list(freq_data[topic].keys()))
        selected_outlet = st.selectbox(f"언론사 선택 ({topic})", options=available_outlets, key=f"wc_outlet_{topic}")
        
        if selected_outlet:
            try:
                wordcloud = WordCloud(
                    font_path='SeoulHangangL.ttf', 
                    width=800, height=400, background_color='white', max_words=100, collocations=False
                ).generate_from_frequencies(freq_data[topic][selected_outlet])
                st.image(wordcloud.to_array(), caption=f"'{selected_outlet}'의 '{topic}' 관련 주요 키워드")
            except FileNotFoundError:
                st.error("폰트 파일(SeoulHangangL.ttf)을 찾을 수 없습니다. 워드클라우드를 표시할 수 없습니다.")

def display_stacked_bar(topic, sentiment_df):
    st.subheader(f"논조 유형별 기사 비율: '{topic}'")
    df_topic = sentiment_df[sentiment_df['topic'] == topic].copy()
    if not df_topic.empty:
        bins = [-float('inf'), -0.5, 0.5, float('inf')]
        labels = ['부정적', '중립적', '긍정적']
        df_topic['sentiment_category'] = pd.cut(df_topic['zscore'], bins=bins, labels=labels)
        
        category_counts = df_topic.groupby(['outlet', 'sentiment_category']).size().unstack(fill_value=0)
        total_counts = category_counts.sum(axis=1)
        percentage_df = (category_counts.T / total_counts).T * 100
        
        fig = px.bar(percentage_df, x=percentage_df.index, y=labels,
                     title=f"'{topic}' 주제에 대한 언론사별 논조 유형 비율 (%)",
                     labels={'value': '기사 비율 (%)', 'outlet': '언론사'},
                     color_discrete_map={'부정적': '#d62728', '중립적': '#7f7f7f', '긍정적': '#2ca02c'},
                     category_orders={"variable": labels})
        st.plotly_chart(fig, use_container_width=True)

def display_topic_distribution_comparison(sentiment_df):
    """주제를 선택하면 해당 주제에 대한 언론사별 논조 분포를 비교합니다."""
    topic_to_compare = st.selectbox("비교할 주제를 선택하세요:", sentiment_df['topic'].unique())
    if topic_to_compare:
        df_topic = sentiment_df[sentiment_df['topic'] == topic_to_compare]
        
        fig = go.Figure()
        for outlet in df_topic['outlet'].unique():
            outlet_scores = df_topic[df_topic['outlet'] == outlet]['zscore']
            counts, bins = np.histogram(outlet_scores, bins=20, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            fig.add_trace(go.Scatter(x=bin_centers, y=counts, mode='lines', name=outlet))
            
        fig.update_layout(
            title=f"'{topic_to_compare}' 주제에 대한 언론사별 Z-score 분포",
            xaxis_title="Z-score (논조의 상대적 강도)",
            yaxis_title="기사 밀도 (Density)",
            legend_title="언론사"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_outlet_distribution_comparison(sentiment_df):
    """언론사를 선택하면 해당 언론사의 주제별 논조 분포를 비교합니다."""
    outlet_to_compare = st.selectbox("비교할 언론사를 선택하세요:", sentiment_df['outlet'].unique())
    if outlet_to_compare:
        df_outlet = sentiment_df[sentiment_df['outlet'] == outlet_to_compare]
        
        fig = go.Figure()
        for topic in df_outlet['topic'].unique():
            topic_scores = df_outlet[df_outlet['topic'] == topic]['zscore']
            counts, bins = np.histogram(topic_scores, bins=20, range=(-3, 3), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            fig.add_trace(go.Scatter(x=bin_centers, y=counts, mode='lines', name=topic))

        fig.update_layout(
            title=f"'{outlet_to_compare}'의 주제별 Z-score 분포",
            xaxis_title="Z-score (논조의 상대적 강도)",
            yaxis_title="기사 밀도 (Density)",
            legend_title="주제"
        )
        st.plotly_chart(fig, use_container_width=True)


# ====================================================================================

sentiment_df, freq_data, pca_df = load_data()

if sentiment_df is None:
    st.stop()

# --- 1. 서론 ---
st.markdown(load_markdown("content/0_introduction.md"), unsafe_allow_html=True)

# 실제 코드 제시
with st.expander("참고: OpenAI API 활용 코드", expanded=False):
    st.markdown(load_markdown("content/gpt.md"), unsafe_allow_html=True)

st.markdown("---")


# --- 2. 종합 분석 ---
st.markdown(load_markdown("content/1_overall_analysis.md"), unsafe_allow_html=True)

if pca_df is not None:
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', text=pca_df.index, title="주제별 논조를 종합한 언론사 포지셔닝 맵")
    fig_pca.update_traces(textposition='top center')
    fig_pca.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey")
    fig_pca.add_vline(x=0, line_width=1, line_dash="dot", line_color="grey")
    st.plotly_chart(fig_pca, use_container_width=True)

pivot_df_heatmap = sentiment_df.pivot_table(index='topic', columns='outlet', values='zscore', aggfunc='mean')
fig_heatmap = px.imshow(pivot_df_heatmap, text_auto=".2f", aspect='auto', color_continuous_scale='RdBu_r', title="주제별/언론사별 평균 Z-score 히트맵 (클릭하여 기사 확인)")
selected_points = plotly_events(fig_heatmap, click_event=True, key="heatmap_click")

if selected_points:
    clicked_topic, clicked_outlet = selected_points[0]['y'], selected_points[0]['x']
    with st.container(border=True):
        st.info(f"**'{clicked_outlet}'**의 **'{clicked_topic}'** 관련 상위 기사 5개 (부정적 논조 순)")
        subset_df = sentiment_df[(sentiment_df['topic'] == clicked_topic) & (sentiment_df['outlet'] == clicked_outlet)].sort_values('zscore')
        for _, row in subset_df.head(5).iterrows():
            st.markdown(f"- [{row['title']}]({row['url']}) (Z-Score: {row['zscore']:.3f})")

st.markdown("---")

# --- 3. 주제별 심층 분석 ---
st.markdown("## 주제별 심층 분석")

topic_map = { '탈원전': '2_탈원전.md', '전장연': '3_전장연.md', '노조파업': '4_노조파업.md' }
for topic, md_filename in topic_map.items():
    st.markdown(load_markdown(f"content/{md_filename}"), unsafe_allow_html=True)
    with st.expander(f"'{topic}' 관련 데이터 시각화 자세히 보기"):
        display_wordcloud(topic, freq_data)
        st.markdown("---")
        display_stacked_bar(topic, sentiment_df)
    st.markdown("---")

# --- 4. 결론 ---
st.markdown("## 결론")

# Finding 1: 이념적 군집
st.markdown(load_markdown("content/5_conclusion_finding1.md"), unsafe_allow_html=True)
display_topic_distribution_comparison(sentiment_df)

# Finding 2: 사안별 상이성
st.markdown(load_markdown("content/5_conclusion_finding2.md"), unsafe_allow_html=True)
display_outlet_distribution_comparison(sentiment_df)

# 최종 제언
st.markdown(load_markdown("content/6_final_recommendation.md"), unsafe_allow_html=True)