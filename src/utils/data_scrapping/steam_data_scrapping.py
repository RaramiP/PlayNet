import requests
import pandas as pd
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")


class SteamScraper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://store.steampowered.com/api/appdetails"
        self.app_list_url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
        self.data_file = "steam_games_dataset.csv"

    def get_all_app_ids(self, start_appid=0):
        """Fetches the list of GAMES only (no DLCs) starting from start_appid."""
        print("Fetching all AppIDs via IStoreService...")

        apps = []
        last_appid = start_appid
        more_items = True

        while more_items:
            params = {
                "key": self.api_key,
                "include_games": "true",
                "include_dlc": "false",
                "include_software": "false",
                "max_results": 10000,
                "last_appid": last_appid,
            }

            try:
                response = requests.get(self.app_list_url, params=params)
                data = response.json()

                if "response" not in data or "apps" not in data["response"]:
                    print("Error: API returned unexpected format.")
                    break

                batch = data["response"]["apps"]
                if not batch:
                    more_items = False
                else:
                    apps.extend(batch)
                    last_appid = batch[-1]["appid"]
                    print(f"Fetched {len(apps)} games so far...")
                    time.sleep(1)

            except Exception as e:
                print(f"Connection error: {e}")
                break

        print(f"Total games found: {len(apps)}")
        return apps

    def get_game_details(self, app_id):
        """Fetches detailed metadata (genres, screenshots)."""
        params = {"appids": app_id}
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if data and str(app_id) in data and data[str(app_id)]["success"]:
                game_data = data[str(app_id)]["data"]

                if game_data.get("type") != "game":
                    return None

                return {
                    "appid": app_id,
                    "name": game_data.get("name"),
                    "genres": [g["description"] for g in game_data.get("genres", [])],
                    "header_image": game_data.get("header_image"),
                    "screenshots": [
                        s["path_thumbnail"] for s in game_data.get("screenshots", [])
                    ][:3],
                    "short_description": game_data.get("short_description"),
                }
            return None

        except Exception as e:
            print(f"Error fetching {app_id}: {e}")
            return None

    def run(self, start_appid=0, limit=1000):
        all_apps = self.get_all_app_ids(start_appid=start_appid)

        if not all_apps:
            print("Could not retrieve game list. Check your API Key.")
            return

        collected_data = []

        if os.path.exists(self.data_file):
            df_existing = pd.read_csv(self.data_file)
            processed_ids = set(df_existing["appid"].tolist())
            print(f"Resuming... {len(processed_ids)} games already scraped.")
        else:
            processed_ids = set()

        count = 0
        for app in all_apps:
            if count >= limit:
                break

            app_id = app["appid"]
            if app_id in processed_ids:
                continue

            print(f"Processing {app.get('name', 'Unknown')} ({app_id})...")
            details = self.get_game_details(app_id)

            if details:
                collected_data.append(details)
                count += 1

            time.sleep(1.5)

            if count % 50 == 0 and collected_data:
                self.save_to_csv(collected_data)
                collected_data = []

        if collected_data:
            self.save_to_csv(collected_data)

    def save_to_csv(self, new_data):
        df = pd.DataFrame(new_data)
        mode = "a" if os.path.exists(self.data_file) else "w"
        header = not os.path.exists(self.data_file)

        df["genres"] = df["genres"].apply(json.dumps)
        df["screenshots"] = df["screenshots"].apply(json.dumps)

        df.to_csv(self.data_file, mode=mode, header=header, index=False)
        print(f"Saved {len(new_data)} records to {self.data_file}")


if __name__ == "__main__":
    scraper = SteamScraper(api_key=API_KEY)
    scraper.run(start_appid=0, limit=2000)
