```python

import time

client = openai.OpenAI(api_key="your-api-key") 

def get_sentiment_score(topic, content):
    prompt = f"""다음 기사는 '{topic}'에 대한 내용입니다.
해당 기사에서 '{topic}'에 대해 나타나는 감정을 -1부터 1 사이의 점수로 수치화해 주세요.
-1은 매우 부정적, 0은 중립, 1은 매우 긍정적입니다. 숫자만 답해 주세요.

{content}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 감정분석 전문가야. 감정 점수를 숫자로 정밀하게 판단해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        score = response.choices[0].message.content.strip()
        return float(score)
    except Exception as e:
        print(f"Error: {e}")
        return None

    df["감정점수"] = None

for i, row in df.iterrows():
    topic = row["주제"] if "주제" in df.columns else "이 기사"
    content = row["본문"]
    score = get_sentiment_score(topic, content)
    df.at[i, "감정점수"] = score

    if i % 50 == 0:
        print(f"{i}개 처리 완료")
        time.sleep(1)

print("전체 감정 분석 완료")
```
