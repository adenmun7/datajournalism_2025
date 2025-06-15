import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# --- 설정 ---
INPUT_FILE = "sentiment.csv"
OUTPUT_FILE = "sentiment_with_titles_final_reused_driver.csv"
MAX_REQUESTS_WORKERS = 15
MAX_SELENIUM_WORKERS = 6 # 동시에 실행할 최대 브라우저 수

# 처리할 4개 신문사 최종 설정
PRESS_CONFIG = {
    '중앙일보': {'method': 'requests', 'selectors': ['#article_title', 'h1.headline']},
    '한겨레': {'method': 'requests', 'selectors': ['h3[class*="ArticleDetailView_title"]', 'h1.title', 'div.article_text h3.title']},
    '오마이뉴스': {'method': 'requests', 'selectors': ['h2.article_tit a', 'h3.tit_view']},
    '조선일보': {'method': 'selenium', 'selectors': ['h1[class*="article-header__headline"] span', 'h1.article-header__headline']}
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def find_title_from_soup(soup, selectors):
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
    return "제목 없음 (선택자 실패)"

def fetch_with_requests(url, selectors):
    """requests 작업을 수행하는 스레드 워커"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        return find_title_from_soup(soup, selectors)
    except Exception as e:
        return f"제목 없음 (Requests 에러: {e.__class__.__name__})"

def selenium_worker(driver_queue, url, selectors):
    """드라이버 풀에서 드라이버를 빌리고 반납하는 스레드 워커"""
    driver = None
    try:
        # 1. 드라이버 풀(Queue)에서 쉬고 있는 드라이버를 가져옴 (없으면 대기)
        driver = driver_queue.get()
        driver.get(url)
        time.sleep(3) # 페이지 로딩 대기
        soup = BeautifulSoup(driver.page_source, 'lxml')
        return find_title_from_soup(soup, selectors)
    except Exception as e:
        return f"제목 없음 (Selenium 에러: {e.__class__.__name__})"
    finally:
        # 2. 작업이 끝나면 (성공하든 실패하든) 드라이버를 다시 풀에 반납
        if driver:
            driver_queue.put(driver)

def main():
    df = pd.read_csv(INPUT_FILE)
    df['제목'] = "처리 안함"
    
    target_presses = list(PRESS_CONFIG.keys())
    df_target = df[df['언론사'].isin(target_presses)].copy()

    df_requests_list = [row for _, row in df_target.iterrows() if PRESS_CONFIG[row['언론사']]['method'] == 'requests']
    df_selenium_list = [row for _, row in df_target.iterrows() if PRESS_CONFIG[row['언론사']]['method'] == 'selenium']
    
    print(f"총 {len(df_target)}개 중 'requests'로 {len(df_requests_list)}개, 'selenium'으로 {len(df_selenium_list)}개를 처리합니다.")
    
    # --- requests 병렬 처리 (변경 없음) ---
    print(f"\n[requests] {len(df_requests_list)}개 작업을 최대 {MAX_REQUESTS_WORKERS}개 스레드로 병렬 처리 시작...")
    # ... (이전 코드와 동일)

    # --- selenium 병렬 처리 (드라이버 풀 사용) ---
    print(f"\n[selenium] {len(df_selenium_list)}개 작업을 최대 {MAX_SELENIUM_WORKERS}개 브라우저로 병렬 처리 시작...")
    if df_selenium_list:
        driver_pool = queue.Queue()
        # 미리 정해진 수만큼 드라이버를 생성하여 풀에 추가
        print(f"  드라이버 풀 생성을 위해 {MAX_SELENIUM_WORKERS}개의 브라우저를 초기화합니다...")
        for _ in range(MAX_SELENIUM_WORKERS):
            options = webdriver.ChromeOptions()
            options.add_argument('--headless'); options.add_argument('--log-level=3'); options.add_argument(f"user-agent={HEADERS['User-Agent']}")
            service = Service(ChromeDriverManager().install())
            driver_pool.put(webdriver.Chrome(service=service, options=options))

        with ThreadPoolExecutor(max_workers=MAX_SELENIUM_WORKERS) as executor:
            future_to_index = {
                executor.submit(selenium_worker, driver_pool, row['URL'], PRESS_CONFIG[row['언론사']]['selectors']): row.name
                for row in df_selenium_list
            }
            for i, future in enumerate(as_completed(future_to_index)):
                index = future_to_index[future]
                df.loc[index, '제목'] = future.result()
                print(f"  (selenium {i+1}/{len(df_selenium_list)}) [{df.loc[index, '언론사']}] 처리 완료")
        
        # 모든 작업이 끝나면 풀에 있는 드라이버들을 모두 종료
        print("  모든 Selenium 작업을 마치고 드라이버를 종료합니다...")
        while not driver_pool.empty():
            driver = driver_pool.get()
            driver.quit()

    print("[selenium] 병렬 처리 완료!")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("-" * 50)
    print("모든 크롤링 완료!")
    print(f"결과가 '{OUTPUT_FILE}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    # requests 처리 부분을 생략하지 않으려면 아래 주석을 해제하세요.
    # main() 함수는 데모를 위해 일부만 실행되도록 수정될 수 있습니다.
    # 실제 실행을 위해서는 전체 main 함수 로직을 사용해야 합니다.
    # 이 코드 블록은 설명을 위해 단순화되어 있습니다.
    
    # 실제 실행 시에는 아래와 같이 main()을 호출하면 됩니다.
    main()