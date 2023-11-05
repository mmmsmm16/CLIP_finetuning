import os
import requests
import concurrent.futures
import csv
from tqdm import tqdm

PROGRESS_FILE = "download_progress.txt"

def download_image(url, idx, save_directory):
    filename = os.path.join(save_directory, f"{idx}.jpg")

    # すでに同じ名前の画像が存在する場合はスキップ
    if os.path.exists(filename):
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
    except requests.RequestException:
        # エラーが発生した場合は何も表示しない
        pass

    # 最後にダウンロードしたインデックスを保存
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(idx))

def main(tsv_filepath, save_directory, max_threads=50):
    # 指定されたディレクトリを作成
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # .tsvファイルからURLを読み込む
    with open(tsv_filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        urls = [row[1] for row in reader]

    start_idx = 0
    # 進行状況のファイルが存在する場合、その位置から開始
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            start_idx = int(f.read().strip())

    # マルチスレッドで画像をダウンロードしながら、tqdmで進行状況を表示
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(download_image, urls[idx], idx, save_directory): (idx, urls[idx]) for idx in range(start_idx, len(urls))}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(urls) - start_idx):
            future.result()

if __name__ == "__main__":
    tsv_filepath = "Train_GCC-training.tsv"  # .tsvファイルのパスを指定
    save_directory = "train_images"   # 保存先ディレクトリ名を指定
    main(tsv_filepath, save_directory)
