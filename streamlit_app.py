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
st.set_page_config(
    page_title="감정분석을 활용한 미디어 편향성 분석",
    layout="wide"
)
st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)


# 데이터 준비
@st.cache_data
def load_sentiment_data():
    try:
        df = pd.read_csv('sentiment.csv')
        df.rename(columns={'주제': 'topic', '언론사': 'outlet', '제목': 'title', '감정점수': 'sentiment', 'zscore': 'zscore', 'URL': 'url'}, inplace=True)
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

st.title("감정분석을 통한 미디어 편향성 분석")
st.markdown("### 데이터저널리즘 : 6팀")
st.markdown("---")


# ### 소개 섹션 ###
with st.container(border=True):
    st.header("프로젝트 소개")
    st.markdown("""
    본 프로젝트는 우리가 가진 언론사에 대한 직관적인 분류가 실제 데이터상에서 어떻게 나타나는지 확인하고자 기획되었습니다. \n
    이를 위해 특정 사회적 갈등 사안에 대한 언론사별 '논조'를 데이터로 수치화하고, 다양한 시각화를 통해 그 차이와 패턴을 분석해 보았습니다.
    """)

    st.subheader("분석 데이터 개요")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(height=200, border=True):
            st.markdown("<p style='text-align: center;'>분석 주제</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>3개</h2>", unsafe_allow_html=True)
            st.caption("<p style='text-align: center; line-height: 1.5;'>탈원전<br>전국장애인차별철폐연대(전장연)<br>노조파업</p>", unsafe_allow_html=True)

    with col2:
        with st.container(height=200, border=True):
            num_outlets = sentiment_df['outlet'].nunique()
            st.markdown("<p style='text-align: center;'>분석 언론사</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{num_outlets}곳</h2>", unsafe_allow_html=True)
            # 언론사 목록을 작은 글씨로 표시
            outlet_list_str = ", ".join(sentiment_df['outlet'].unique())
            st.caption(f"<p style='text-align: center; line-height: 1.5;'>{outlet_list_str}</p>", unsafe_allow_html=True)

    with col3:
        with st.container(height=200, border=True):
            st.markdown("<p style='text-align: center;'>총 수집 기사</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>6,940개</h2>", unsafe_allow_html=True)
            st.caption("<p style='text-align: center; line-height: 1.5;'>&nbsp;</p>", unsafe_allow_html=True)

    st.markdown("""
    - **1. 감성 분석 (Sentiment Analysis)**
        - ChatGPT API를 활용하여 6,940개 기사 본문 전체의 논조를 분석했습니다.
        - 각 기사는 주제에 대해 얼마나 긍정적/부정적인지에 따라 **-1.0(부정) ~ +1.0(긍정)** 사이의 '감성 점수'를 부여받습니다.
    """)

    st.markdown("""
    - **2. 데이터 표준화 (Standardization)**
        - 주제별로 편향이 존재할 수 있으므로, 단순 감성 점수만으로는 객관적인 비교가 어렵습니다.
        - 각 주제 내에서 기사들의 감성 점수를 Z-score(표준점수)로 변환하면, 논조의 '상대적인 강도'를 동일한 척도상에서 비교할 수 있습니다.
    """)
    with st.expander("Z-score 계산 공식 보기"):
        st.latex(r"Z = \frac{(X - \mu)}{\sigma}")
        st.caption(r"$X$: 개별 기사 점수, $\mu$: 해당 주제의 전체 기사 점수 평균, $\sigma$: 해당 주제의 전체 기사 점수 표준편차")

    st.caption("▼ 실제 데이터 예시 (sentiment.csv)")
    columns_to_show = ['topic', 'outlet', 'title', 'sentiment', 'zscore', 'url']
    st.dataframe(sentiment_df[columns_to_show].head(3), use_container_width=True)
st.markdown("---")
# ### 소개 끝 ###

# ====================================================================================
# ====================================================================================


# 1. wc
st.header("1. 주제별/언론사별 주요 키워드 시각화")

if freq_data:
    max_words_slider = st.slider(
        "표시할 최대 단어 개수를 선택하세요:", 
        min_value=10,
        max_value=200,
        value=30,
        step=10
    )
    # ---------------------------------------------
    
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
            wordcloud = WordCloud(
                font_path='SeoulHangangL.ttf', 
                width=1200, 
                height=800, 
                background_color='white',
                max_words=max_words_slider
            ).generate_from_frequencies(freq_data[selected_topic_wc][selected_outlet_wc])
            
            st.image(wordcloud.to_array(), caption=f"'{selected_outlet_wc}'의 '{selected_topic_wc}' 관련 주요 키워드 (상위 {max_words_slider}개)")
            
        except FileNotFoundError: 
            st.error("폰트 파일이 없습니다.")
        except Exception as e: 
            st.warning(f"워드클라우드 오류: {e}")
else: 
    st.info("빈도수 데이터 로드 불가")

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

# 3. bar
st.header("3. 논조 유형별 기사 비율 분석")
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

# 4. PCA
st.header("4. PCA 분석")
fig_pca = get_pca_figure_from_csv()
if fig_pca:
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.warning("pca_results.csv 파일을 찾을 수 없습니다.")

# 5. heatmap
st.header("5. 상호작용형 히트맵")
n_articles = st.number_input(label="클릭 시 표시할 기사 수를 선택하세요:", min_value=1, max_value=200, value=10, step=1)
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
            annotations.append(go.layout.Annotation(text=f"{value:.2f}", x=outlet, y=topic, xref='x', yref='y', showarrow=False, font=dict(color=font_color, size=15)))
fig_heatmap.update_layout(annotations=annotations, height=700)

selected_points = plotly_events(fig_heatmap, click_event=True, key="heatmap_click")
if selected_points:
    clicked_topic, clicked_outlet = selected_points[0]['y'], selected_points[0]['x']
    clicked_value = pivot_df_heatmap.loc[clicked_topic, clicked_outlet]
    subset_df = sentiment_df[(sentiment_df['topic'] == clicked_topic) & (sentiment_df['outlet'] == clicked_outlet)]
    if not subset_df.empty:
        closest_articles = subset_df.iloc[(subset_df['zscore'] - clicked_value).abs().argsort()[:n_articles]]
        st.subheader(f"'{clicked_outlet}'의 '{clicked_topic}' 관련 기사")
        st.info(f"선택된 셀의 값({clicked_value:.2f})과 근접한 점수의 기사 목록입니다.")
        for _, row in closest_articles.iterrows():
            with st.container(border=True):
                st.markdown(f"**제목**: {row['title']} / [기사 링크]({row['url']}) (Z-Score: {row['zscore']:.3f})")

st.markdown("---")

# 6. graph1
st.header("6. 주제별 언론사 논조 분포 비교")
topic_to_compare = st.selectbox("비교할 주제를 선택하세요:", sentiment_df['topic'].unique())
if topic_to_compare:
    df_topic = sentiment_df[sentiment_df['topic'] == topic_to_compare]
    hist_data = [go.Scatter(x=(b:=(e[:-1]+e[1:])/2), y=c/c.sum() if c.sum()>0 else c, mode='lines', name=o) for o in df_topic['outlet'].unique() for c,e in [np.histogram(df_topic[df_topic['outlet']==o]['zscore'], bins=20)]]
    layout = go.Layout(title=f"'{topic_to_compare}' 주제에 대한 언론사별 z-score 분포", xaxis_title="z-score 구간", yaxis_title="기사 비율 (Density)")
    fig_line1 = go.Figure(data=hist_data, layout=layout)
    st.plotly_chart(fig_line1, use_container_width=True)

st.markdown("---")

# 6. graph2
st.header("6. 언론사별 주제 논조 분포 비교")
outlet_to_compare = st.selectbox("비교할 언론사를 선택하세요:", sentiment_df['outlet'].unique())
if outlet_to_compare:
    df_outlet = sentiment_df[sentiment_df['outlet'] == outlet_to_compare]
    hist_data2 = [go.Scatter(x=(b:=(e[:-1]+e[1:])/2), y=c/c.sum() if c.sum()>0 else c, mode='lines', name=t) for t in df_outlet['topic'].unique() for c,e in [np.histogram(df_outlet[df_outlet['topic']==t]['zscore'], bins=20, range=(-3,3))]]
    layout2 = go.Layout(title=f"'{outlet_to_compare}'의 주제별 z-score 분포", xaxis_title="z-score 구간", yaxis_title="기사 비율 (Density)")
    fig_line2 = go.Figure(data=hist_data2, layout=layout2)
    st.plotly_chart(fig_line2, use_container_width=True)

st.markdown("---")

# 7. conclusion
# ====================================================================================
# --- 최종 결론 섹션 (텍스트 중심) ---
# ====================================================================================

st.markdown("---")
st.header("결론")

st.markdown("""
본 프로젝트는 '보수', '진보'라는 우리의 직관적인 언론 분류가 실제 데이터에서 어떻게 나타나는지 확인하는 것에서 시작했습니다. 
분석 결과, 언론 지형은 단순한 이분법을 넘어 더 복잡하고 다층적인 모습을 보여주었습니다.
""")

with st.container(border=True):
    st.subheader("1: 큰 틀에서는 뚜렷한 '이념적 군집'이 존재한다")
    st.markdown("""
    **PCA 분석**에서 3가지 주제에 대한 모든 언론사의 논조를 종합했을 때, 지도상에서 크게 두 개의 그룹으로 나뉘고 있습니다. **히트맵**에서도 어느 정도의 경향성을 확인할 수 있습니다.
                
    이는 우리가 흔히 말하는 정치적 편향성의 구도가 데이터상에서 실재함을 보여주고 있습니다.
    """)

with st.container(border=True):
    st.subheader("2: 그러나 모든 사안에 동일한 잣대를 들이대지 않는다")
    st.markdown("""
    하지만 **상호작용형 히트맵**을 통해 더 깊이 들여다보면, 언론사들이 모든 이슈에 대해 기계적으로 같은 수준의 편향성을 보이지는 않는다는 점을 발견할 수 있습니다.
    
    만약 한 언론사가 모든 사안에 동일한 강도의 논조를 보인다면, 히트맵의 한 행은 모두 같은 색으로 칠해져야 합니다. 하지만 실제로는 특정 언론사가 어떤 주제에는 매우 강한 긍정/부정(진한 색)을 보이지만, 어떤 주제에는 비교적 중립에 가까운(옅은 색) 태도를 보입니다. 
    
    이는 언론사가 가진 **기본적인 정치적 편향성과는 별개로, 개별 사안의 특성에 따라 논조의 '강도'와 '선명성'을 조절**하고 있음을 시사합니다.
    """)


# --- 최종 제언 ---
st.subheader("최종 제언: '프레임'을 읽는 미디어 리터러시")

st.success("""
**언론은 하나의 이념이나 정치적 선호에 완전히 구속되기보다는, 그것에 어느 정도만 호응하며 스스로 담론을 만들어내고 있습니다.**

언론사들은 확실히 구별되는 진영을 구축하지만, 동시에 모든 사회 현상을 그것에만 의존해 서술하지는 않습니다. 각 사안의 복잡성 위에서 **사안별로 다른 '프레임'을 선택한다는 것입니다.**

언론에 대해 **'보수', '진보'등과 같이 고정된 라벨로만 바라보는 시각을 넘어,** 이 데이터가 보여주듯 **'어떤 사안에 대해, 어떤 근거로, 어떤 어휘를 사용해'** 특정 논조를 형성하는지를 비판적으로 살펴보는 미디어 리터러시가 필요하다고 하겠습니다. 
           
본 프로젝트가 독자들이 주체적으로 정보를 소비하는 데 작게나마 도움이 되기를 바랍니다.
""")