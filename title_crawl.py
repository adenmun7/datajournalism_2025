import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

INPUT_FILE = "sentiment.csv"
OUTPUT_FILE = "sentiment_with_titles_final_reused_driver.csv"
MAX_REQUESTS_WORKERS = 15
MAX_SELENIUM_WORKERS = 6

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
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        return find_title_from_soup(soup, selectors)
    except Exception as e:
        return f"제목 없음 (Requests 에러: {e.__class__.__name__})"

def selenium_worker(driver_queue, url, selectors):
    driver = None
    try:
        driver = driver_queue.get()
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        return find_title_from_soup(soup, selectors)
    except Exception as e:
        return f"제목 없음 (Selenium 에러: {e.__class__.__name__})"
    finally:
        if driver:
            driver_queue.put(driver)

def main():
    df = pd.read_csv(INPUT_FILE)
    df['제목'] = "처리 안함"
    
    target_presses = list(PRESS_CONFIG.keys())
    df_target = df[df['언론사'].isin(target_presses)].copy()

    df_requests_list = [row for _, row in df_target.iterrows() if PRESS_CONFIG[row['언론사']]['method'] == 'requests']
    df_selenium_list = [row for _, row in df_target.iterrows() if PRESS_CONFIG[row['언론사']]['method'] == 'selenium']

    #
    
    if df_selenium_list:
        driver_pool = queue.Queue()
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
        
        while not driver_pool.empty():
            driver = driver_pool.get()
            driver.quit()

    print("처리 완료")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("크롤링 완료")

if __name__ == "__main__":
    main()