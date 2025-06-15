import pandas as pd
import os

def merge_crawled_files_safely():
    """
    파일 목록을 먼저 확인하고, 3개의 CSV 파일을 안전하게 병합합니다.
    """
    # 1. 현재 폴더의 파일 목록을 출력하여 파일 존재 여부와 정확한 이름을 확인합니다.
    try:
        print("현재 폴더에 있는 파일 목록:")
        file_list_in_dir = os.listdir('.')
        for f in file_list_in_dir:
            print(f"- {f}")
        print("-" * 50)
    except Exception as e:
        print(f"파일 목록을 읽는 중 에러 발생: {e}")
        return

    # 2. 병합할 파일들의 이름을 리스트로 정의합니다.
    files_to_merge = [
        "sentiment_동아일보_경향신문_연합뉴스.csv",
        "sentiment_조선일보.csv",
        "sentiment_중앙일보_한겨레_오마이뉴스.csv"
    ]
    
    dataframes = {}
    all_files_loaded = True

    # 3. 각 파일을 개별적으로 불러옵니다.
    print("파일을 하나씩 불러옵니다...")
    for filename in files_to_merge:
        try:
            dataframes[filename] = pd.read_csv(filename)
            print(f"[성공] '{filename}' 파일을 불러왔습니다.")
        except FileNotFoundError:
            print(f"[오류] '{filename}' 파일을 찾을 수 없습니다. 파일 이름과 업로드 상태를 확인해주세요.")
            all_files_loaded = False
    
    # 만약 파일 하나라도 불러오지 못했다면, 작업을 중단합니다.
    if not all_files_loaded:
        print("\n필요한 파일을 모두 불러오지 못했으므로 병합 작업을 중단합니다.")
        return

    # 4. 모든 파일을 성공적으로 불러왔다면, 병합을 시작합니다.
    print("\n모든 파일을 성공적으로 불러왔습니다. 병합을 시작합니다...")
    
    # 첫 번째 데이터프레임을 기준으로 삼습니다.
    df_final = dataframes[files_to_merge[0]].copy()

    # 두 번째, 세 번째 파일의 내용을 순서대로 덮어씁니다.
    for filename in files_to_merge[1:]:
        df_to_merge = dataframes[filename]
        processed_rows = (df_to_merge['제목'] != '처리 안함') & (df_to_merge['제목'].notna())
        df_final.loc[processed_rows, '제목'] = df_to_merge.loc[processed_rows, '제목']
        print(f"- '{filename}'에서 {processed_rows.sum()}개의 제목을 병합했습니다.")

    # 5. 최종 결과를 새 파일로 저장합니다.
    output_filename = "sentiment_final_merged.csv"
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

    # 6. 완료 메시지와 요약을 출력합니다.
    total_processed = (df_final['제목'] != '처리 안함') & (df_final['제목'].notna())
    print("\n" + "="*50)
    print("파일 병합이 성공적으로 완료되었습니다!")
    print(f"최종 결과가 '{output_filename}' 파일에 저장되었습니다.")
    print(f"총 {len(df_final)}개의 행 중 {total_processed.sum()}개의 제목이 채워졌습니다.")
    print("="*50)

# 함수 실행
merge_crawled_files_safely()