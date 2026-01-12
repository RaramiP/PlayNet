
import os
import ast
import pandas as pd

# Genres to keep (everything else becomes "Other")
KEEP_GENRES = [
    'Action', 'Free To Play', 'Strategy', 'Adventure', 'Indie', 'RPG',
    'Casual', 'Simulation', 'Racing', 'Massively Multiplayer', 'Sports'
]

FINAL_GENRES = KEEP_GENRES + ['Other']


def parse_genres(genres_str: str) -> list[str]:
    """Parse the genres string from CSV"""
    try:
        return ast.literal_eval(genres_str)
    except Exception:
        return []


def remap_genres(original_genres: list[str]) -> dict[str, int]:
    """Convert genre list to multi-hot dict with Other category."""
    result = {genre: 0 for genre in FINAL_GENRES}
    
    for genre in original_genres:
        if genre in KEEP_GENRES:
            result[genre] = 1
        else:
            result['Other'] = 1
    
    return result


def generate_dataset_csv(steam_csv: str, img_dir: str, output_csv: str):
    """Generate training CSV from Steam data and image directory."""
    
    # Load Steam data
    steam_df = pd.read_csv(steam_csv)
    print(f"Loaded {len(steam_df)} games from Steam CSV")
    
    # Create lookup by appid
    steam_lookup = {row['appid']: row for _, row in steam_df.iterrows()}
    
    records = []
    missing_games = []
    
    # Iterate through image folders
    for folder_name in sorted(os.listdir(img_dir)):
        folder_path = os.path.join(img_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Parse app_id from folder name (e.g., "10_Counter-Strike" -> 10)
        try:
            app_id = int(folder_name.split('_')[0])
        except ValueError:
            print(f"Skipping invalid folder: {folder_name}")
            continue
        
        # Look up in Steam data
        if app_id not in steam_lookup:
            missing_games.append(folder_name)
            continue
        
        game_data = steam_lookup[app_id]
        original_genres = parse_genres(game_data['genres'])
        genre_labels = remap_genres(original_genres)
        
        # Add record for each image
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                record = {
                    'image_path': f"{folder_name}/{img_name}",
                    'app_id': app_id,
                    'game': game_data['name'],
                    **genre_labels
                }
                records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"\nGenerated {output_csv} with {len(df)} images from {df['app_id'].nunique()} games")
    
    if missing_games:
        print(f"\nWarning: {len(missing_games)} game folders not found in Steam CSV:")
        for g in missing_games[:10]:
            print(f"  - {g}")
        if len(missing_games) > 10:
            print(f"  ... and {len(missing_games) - 10} more")
    
    print(f"\nGenre distribution:")
    print(df[FINAL_GENRES].sum().sort_values(ascending=False).to_string())
    
    return df

df = generate_dataset_csv(
    steam_csv="../data/info.csv",
    img_dir="../data/dataset_images",
    output_csv="../data/dataset.csv"
)