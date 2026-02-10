import pandas as pd
import requests
import os
import json
import ast
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 

INPUT_CSV = "data/steam_games_dataset.csv"  
OUTPUT_DIR = "data/dataset_images"          
MAX_WORKERS = 10                       

def download_image(url, save_path):
    """Helper function to download a single image."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception:
        return False
    return False

def process_row(row):
    """Processes one game: parses URLs and downloads images."""
    app_id = row['appid']
    game_name = str(row['name']).replace(':', '').replace('/', '').replace('\\', '').strip()  # Clean filename
    
    game_folder = os.path.join(OUTPUT_DIR, f"{app_id}_{game_name}")
    
    try:
        try:
            urls = json.loads(row['screenshots'])
        except:
            urls = ast.literal_eval(row['screenshots'])
            
        if not isinstance(urls, list) or not urls:
            return
            
    except Exception as e:
        print(f"Could not parse screenshots for {app_id}")
        return

    os.makedirs(game_folder, exist_ok=True)

    for i, url in enumerate(urls):
        url = url.strip()
        
        file_name = f"screenshot_{i}.jpg"
        save_path = os.path.join(game_folder, file_name)
        
        if os.path.exists(save_path):
            continue
            
        download_image(url, save_path)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}")
        return

    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV)
    
    # Create main output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Found {len(df)} games. Starting download with {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=len(df)))
        
    print("\nDownload complete!")

if __name__ == "__main__":
    main()