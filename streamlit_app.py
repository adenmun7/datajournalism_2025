import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import glob
from streamlit_plotly_events import plotly_events
import os

# 기본 설정
st.set_page_config(page_title="미디어 편향성 분석")

# 데이터 준비
@st.cache_data
def load_sentiment_data():
    try:
        df = pd.read_csv('sentiment.csv')
        df.rename(columns={'주제': 'topic', '언론사': 'outlet', '감정점수': 'sentiment', 'zscore': 'zscore', 'URL': 'url'}, inplace=True)
        return df
    except FileNotFoundError: return None

@st.cache_data
def load_statistics_data():
    try: return pd.read_csv('statistics.csv')
    except FileNotFoundError: return None

@st.cache_data
def load_frequency_data(folder_path):
    freq_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not freq_files: return None
    data_dict = {}
    for file_path in freq_files:
        try:
            filename = os.path.basename(file_path).replace('.csv', '')
            parts = filename.split('_')
            if len(parts) == 2:
                topic, outlet = parts[0], parts[1]
                df = pd.read_csv(file_path)
                freq_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                if topic not in data_dict: data_dict[topic] = {}
                data_dict[topic][outlet] = freq_dict
        except Exception: continue
    return data_dict

@st.cache_data
def get_pca_figure_from_csv():
    try:
        pca_df = pd.read_csv('pca_results.csv', index_col='outlet') # index_col 설정 중요
        
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', text=pca_df.index, title="주제별 논조를 종합한 언론사 포지셔닝 맵")
        fig_pca.update_traces(textposition='top center')
        fig_pca.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey")
        fig_pca.add_vline(x=0, line_width=1, line_dash="dot", line_color="grey")
        return fig_pca
    except FileNotFoundError:
        return None

sentiment_df = load_sentiment_data()
stats_df = load_statistics_data()
freq_data = load_frequency_data('frequency_csvs')

if sentiment_df is None:
    st.error("`sentiment.csv` 파일을 찾을 수 없습니다.")
    st.stop()

# ====================================================================================
# ====================================================================================
st.title("텍스트 마이닝 기반 미디어 분석")
st.markdown("---")

# 1. wc
st.header("1. 주제별/언론사별 주요 키워드 시각화")
if freq_data:
    topic_list = sorted(list(freq_data.keys()))
    col1, col2 = st.columns(2)
    with col1:
        selected_topic_wc = st.selectbox("주제 선택 (워드클라우드)", options=topic_list)
    with col2:
        if selected_topic_wc:
            available_outlets_wc = sorted(list(freq_data.get(selected_topic_wc, {}).keys()))
            selected_outlet_wc = st.selectbox("언론사 선택 (워드클라우드)", options=available_outlets_wc)
    if selected_topic_wc and selected_outlet_wc and freq_data.get(selected_topic_wc, {}).get(selected_outlet_wc):
        try:
            wordcloud = WordCloud(font_path='SeoulHangangL.ttf', width=1200, height=800, background_color='white').generate_from_frequencies(freq_data[selected_topic_wc][selected_outlet_wc])
            st.image(wordcloud.to_array(), caption=f"'{selected_outlet_wc}'의 '{selected_topic_wc}' 관련 주요 키워드")
        except FileNotFoundError: st.error("폰트 파일(`SeoulHangangL.ttf`)이 없습니다.")
        except Exception as e: st.warning(f"워드클라우드 오류: {e}")
else: st.info("빈도수 데이터 로드 불가")

st.markdown("---")

# 2. basic stats
st.header("2. 기사 수 및 감성점수 통계")
if stats_df is not None:
    st.markdown("언론사별, 주제별로 수집된 기사의 수와 감성점수의 평균 및 표준편차입니다.")
    fig_stats = px.bar(stats_df, x='언론사', y='기사수', color='주제', barmode='group', title="주제별/언론사별 기사 수")
    st.plotly_chart(fig_stats, use_container_width=True)
    st.dataframe(stats_df)
else: st.info("`statistics.csv` 파일이 없습니다.")

st.markdown("---")

# 3. heatmap
st.header("3. 상호작용형 히트맵")
n_articles = st.number_input(label="클릭 시 표시할 기사 수를 선택하세요:", min_value=1, max_value=20, value=3, step=1)
pivot_df_heatmap = sentiment_df.pivot_table(index='topic', columns='outlet', values='zscore', aggfunc='mean')
fig_heatmap = px.imshow(pivot_df_heatmap, aspect='auto', color_continuous_scale='RdBu_r', title="평균 z-score 히트맵")

annotations = []
z_min, z_max = pivot_df_heatmap.min().min(), pivot_df_heatmap.max().max()
z_mid = (z_min + z_max) / 2
for topic in pivot_df_heatmap.index:
    for outlet in pivot_df_heatmap.columns:
        value = pivot_df_heatmap.loc[topic, outlet]
        if pd.notna(value):
            font_color = 'white' if abs(value - z_mid) > (z_max - z_min) * 0.3 else 'black'
            annotations.append(go.layout.Annotation(text=f"{value:.2f}", x=outlet, y=topic, xref='x', yref='y', showarrow=False, font=dict(color=font_color, size=10)))
fig_heatmap.update_layout(annotations=annotations)

selected_points = plotly_events(fig_heatmap, click_event=True, key="heatmap_click")
if selected_points:
    clicked_topic, clicked_outlet = selected_points[0]['y'], selected_points[0]['x']
    clicked_value = pivot_df_heatmap.loc[clicked_topic, clicked_outlet]
    subset_df = sentiment_df[(sentiment_df['topic'] == clicked_topic) & (sentiment_df['outlet'] == clicked_outlet)]
    if not subset_df.empty:
        closest_articles = subset_df.iloc[(subset_df['zscore'] - clicked_value).abs().argsort()[:n_articles]]
        st.subheader(f"'{clicked_outlet}'의 '{clicked_topic}' 관련 기사")
        st.info(f"선택된 셀의 값({clicked_value:.2f})과 가장 유사한 점수의 기사 목록입니다.")
        for _, row in closest_articles.iterrows():
            with st.container(border=True):
                st.markdown(f"**URL**: [기사 링크]({row['url']}) (점수: {row['zscore']:.3f})")

st.markdown("---")

# 4. graph1
st.header("4. 주제별 언론사 논조 분포 비교")
topic_to_compare = st.selectbox("비교할 주제를 선택하세요:", sentiment_df['topic'].unique())
if topic_to_compare:
    df_topic = sentiment_df[sentiment_df['topic'] == topic_to_compare]
    hist_data = [go.Scatter(x=(b:=(e[:-1]+e[1:])/2), y=c/c.sum() if c.sum()>0 else c, mode='lines', name=o) for o in df_topic['outlet'].unique() for c,e in [np.histogram(df_topic[df_topic['outlet']==o]['zscore'], bins=20)]]
    layout = go.Layout(title=f"'{topic_to_compare}' 주제에 대한 언론사별 z-score 분포", xaxis_title="z-score 구간", yaxis_title="기사 비율 (Density)")
    fig_line1 = go.Figure(data=hist_data, layout=layout)
    st.plotly_chart(fig_line1, use_container_width=True)

st.markdown("---")

# 5. graph2
st.header("5. 언론사별 주제 논조 분포 비교")
outlet_to_compare = st.selectbox("비교할 언론사를 선택하세요:", sentiment_df['outlet'].unique())
if outlet_to_compare:
    df_outlet = sentiment_df[sentiment_df['outlet'] == outlet_to_compare]
    hist_data2 = [go.Scatter(x=(b:=(e[:-1]+e[1:])/2), y=c/c.sum() if c.sum()>0 else c, mode='lines', name=t) for t in df_outlet['topic'].unique() for c,e in [np.histogram(df_outlet[df_outlet['topic']==t]['zscore'], bins=20, range=(-3,3))]]
    layout2 = go.Layout(title=f"'{outlet_to_compare}'의 주제별 z-score 분포", xaxis_title="z-score 구간", yaxis_title="기사 비율 (Density)")
    fig_line2 = go.Figure(data=hist_data2, layout=layout2)
    st.plotly_chart(fig_line2, use_container_width=True)

st.markdown("---")

# 6. bar
st.header("6. 논조 유형별 기사 비율 분석")
topic_for_stacked = st.selectbox("분석할 주제 선택:", sentiment_df['topic'].unique(), key='stacked_topic')
if topic_for_stacked:
    df_stacked = sentiment_df[sentiment_df['topic'] == topic_for_stacked].copy()
    bins, labels = [-float('inf'), -0.5, 0.5, float('inf')], ['부정적', '중립적', '긍정적']
    df_stacked['sentiment_category'] = pd.cut(df_stacked['zscore'], bins=bins, labels=labels)
    category_counts = df_stacked.groupby(['outlet', 'sentiment_category']).size().reset_index(name='count')
    if not category_counts.empty:
        total_counts = category_counts.groupby('outlet')['count'].transform('sum')
        category_counts['percentage'] = (category_counts['count'] / total_counts) * 100
        fig_stacked = px.bar(category_counts, x='outlet', y='percentage', color='sentiment_category', title=f"'{topic_for_stacked}' 주제에 대한 언론사별 논조 유형 비율", labels={'outlet': '언론사', 'percentage': '기사 비율 (%)'}, color_discrete_map={'부정적': '#d62728', '중립적': '#7f7f7f', '긍정적': '#2ca02c'})
        st.plotly_chart(fig_stacked, use_container_width=True)

st.markdown("---")

# 7. PCA
st.header("7. 언론사 포지셔닝 맵 (PCA 분석)")
st.info("모든 주제에 대한 논조를 종합하여 언론사들의 상대적 위치(유사도)를 보여줍니다.")
fig_pca = get_pca_figure_from_csv()
if fig_pca:
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.warning("pca_results.csv 파일을 찾을 수 없습니다.")