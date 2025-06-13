import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import glob
from streamlit_plotly_events import plotly_events
import os


st.set_page_config(page_title="미디어 편향성 분석")


@st.cache_data
def load_sentiment_data():
    try:
        df = pd.read_csv('sentiment.csv')
        df.rename(columns={'주제': 'topic', '언론사': 'outlet', '감정점수': 'sentiment', 'zscore': 'zscore', 'URL': 'url'}, inplace=True)
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_statistics_data():
    try:
        return pd.read_csv('statistics.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def load_frequency_data(folder_path):
    freq_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not freq_files:
        return None
    
    data_dict = {}
    for file_path in freq_files:
        try:
            filename = os.path.basename(file_path)
            base_name = filename.replace('.csv', '')
            parts = base_name.split('_')
            if len(parts) == 2:
                topic, outlet = parts[0], parts[1]
            else:
                continue

            df = pd.read_csv(file_path)
            freq_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            
            if topic not in data_dict:
                data_dict[topic] = {}
            data_dict[topic][outlet] = freq_dict
        
        except Exception as e:
            st.warning(f"파일 '{file_path}' 처리 중 오류: {e}")
            
    return data_dict


###


sentiment_df = load_sentiment_data()
stats_df = load_statistics_data()
freq_data = load_frequency_data('frequency_csvs')

if sentiment_df is None:
    st.error("`sentiment.csv` 파일을 찾을 수 없습니다. 확인 후 다시 실행해주세요.")
    st.stop()

# main
st.title("텍스트 마이닝 기반 미디어 분석")
st.markdown("---")

# 1. wc
st.header("0. 주제별/언론사별 주요 키워드 시각화")
if freq_data:
    col1, col2 = st.columns(2)
    with col1:
        sorted_topics = sorted(list(freq_data.keys()))
        selected_topic_wc = st.selectbox("주제 선택 (워드클라우드)", options=sorted_topics)
    with col2:
        if selected_topic_wc:
            sorted_outlets = sorted(list(freq_data[selected_topic_wc].keys()))
            selected_outlet_wc = st.selectbox("언론사 선택 (워드클라우드)", options=sorted_outlets)

    if selected_topic_wc and selected_outlet_wc:
        try:
            if selected_outlet_wc in freq_data[selected_topic_wc]:
                wordcloud = WordCloud(
                    font_path='SeoulHangangL.ttf',
                    width=1200,
                    height=800,
                    background_color='white'
                ).generate_from_frequencies(freq_data[selected_topic_wc][selected_outlet_wc])
                
                st.image(wordcloud.to_array(), caption=f"'{selected_outlet_wc}'의 '{selected_topic_wc}' 관련 주요 키워드")
            else:
                st.warning("데이터 없음")
        except FileNotFoundError:
            st.error("폰트 파일 누락")
        except Exception as e:
            st.warning(f"워드클라우드 오류: {e}")
else:
    st.info("데이터 로드 불가")

st.markdown("---")

# 2. basic stats
st.header("1. 기사 수 및 감성점수 통계")
if stats_df is not None:
    st.markdown("언론사별, 주제별로 수집된 기사의 수와 감성점수의 평균 및 표준편차입니다.")
    fig_stats = px.bar(
        stats_df,
        x='언론사',
        y='기사수',
        color='주제',
        barmode='group',
        title="주제별/언론사별 기사 수",
        labels={'언론사':'언론사', '기사수':'기사 수', '주제':'주제'}
    )
    st.plotly_chart(fig_stats, use_container_width=True)
    st.dataframe(stats_df)
else:
    st.info("`statistics.csv` 파일이 없습니다.")

st.markdown("---")

# 3. heatmap
st.header("2. 상호작용형 히트맵")
# score_type_heatmap = st.radio("표시할 점수 선택 (히트맵)", ('zscore', '원본 감성점수(sentiment)'), key='heatmap_radio')
score_column_heatmap = 'zscore' # if score_type_heatmap == 'zscore' else 'sentiment'

pivot_df_heatmap = sentiment_df.pivot_table(index='topic', columns='outlet', values=score_column_heatmap, aggfunc='mean')
fig_heatmap = px.imshow(
    pivot_df_heatmap,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdBu_r',
    title=f"평균 {score_column_heatmap} 히트맵"
)
selected_points = plotly_events(fig_heatmap, click_event=True, key="heatmap_click")

if selected_points:
    clicked_topic = selected_points[0]['y']
    clicked_outlet = selected_points[0]['x']
    clicked_value = pivot_df_heatmap.loc[clicked_topic, clicked_outlet]

    subset_df = sentiment_df[(sentiment_df['topic'] == clicked_topic) & (sentiment_df['outlet'] == clicked_outlet)]
    if not subset_df.empty:
        closest_article = subset_df.iloc[(subset_df[score_column_heatmap] - clicked_value).abs().argsort()[:1]]

        st.subheader(f"'{clicked_outlet}'의 '{clicked_topic}' 관련 기사")
        st.info(f"선택된 셀의 값({clicked_value:.2f})과 가장 유사한 점수의 기사입니다.")
        for i, row in closest_article.iterrows():
            st.markdown(f"- **언론사**: {row['outlet']}")
            st.markdown(f"- **URL**: [기사 링크]({row['url']})")
            st.markdown(f"- **점수**: {row[score_column_heatmap]:.3f}")

st.markdown("---")

# 4. graph1
st.header("3. 주제별 언론사 논조 분포 비교")

# score_type_line1 = st.radio("표시할 점수 선택 (언론사 비교)", ('zscore', '원본 감성점수(sentiment)'), key='line1_radio')
score_column_line1 = 'zscore' # if score_type_line1 == 'zscore' else 'sentiment'

topic_to_compare = st.selectbox("비교할 주제를 선택하세요:", sentiment_df['topic'].unique())

if topic_to_compare:
    df_topic = sentiment_df[sentiment_df['topic'] == topic_to_compare]
    
    hist_data = []
    for outlet in df_topic['outlet'].unique():
        df_outlet = df_topic[df_topic['outlet'] == outlet]
        counts, bin_edges = np.histogram(df_outlet[score_column_line1], bins=20)
        
        density = counts / counts.sum() if counts.sum() > 0 else counts
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist_data.append(go.Scatter(x=bin_centers, y=density, mode='lines', name=outlet))

    layout = go.Layout(
        title=f"'{topic_to_compare}' 주제에 대한 언론사별 {score_column_line1} 분포",
        xaxis_title=f"{score_column_line1} 구간",
        yaxis_title="기사 비율 (Density)"
    )
    fig_line1 = go.Figure(data=hist_data, layout=layout)
    st.plotly_chart(fig_line1, use_container_width=True)
    st.info("그래프의 각 선은 해당 언론사의 기사들이 어떤 점수대에 집중되어 있는지 보여줍니다.")

st.markdown("---")

# 5. graph2
st.header("4. 언론사별 주제 논조 분포 비교")

outlet_to_compare = st.selectbox("비교할 언론사를 선택하세요:", sentiment_df['outlet'].unique())

if outlet_to_compare:
    df_outlet = sentiment_df[sentiment_df['outlet'] == outlet_to_compare]
    
    hist_data2 = []
    for topic in df_outlet['topic'].unique():
        df_topic2 = df_outlet[df_outlet['topic'] == topic]
        counts, bin_edges = np.histogram(df_topic2['zscore'], bins=20, range=(-3, 3))
        density = counts / counts.sum() if counts.sum() > 0 else counts
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist_data2.append(go.Scatter(x=bin_centers, y=density, mode='lines', name=topic))

    layout2 = go.Layout(
        title=f"'{outlet_to_compare}'의 주제별 z-score 분포",
        xaxis_title="z-score 구간",
        yaxis_title="기사 비율 (Density)"
    )
    fig_line2 = go.Figure(data=hist_data2, layout=layout2)
    st.plotly_chart(fig_line2, use_container_width=True)
    st.info(f"'{outlet_to_compare}'가 각 주제에 대해 어떤 논조 분포를 보이는지 비교합니다.")


# st.sidebar.info("미디어 편향성 분석 대시보드")
