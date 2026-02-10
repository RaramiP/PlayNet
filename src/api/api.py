from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import shutil

from utils.data_scrapping.steam_data_scrapping import SteamScraper
from utils.data_scrapping.download_image import main as download_images
from utils.data_scrapping.make_dataset import generate_dataset_csv
from model.train import run_training
from model.predict import load_model, preprocess_image, predict, DEVICE

app = FastAPI(title="Steam Game Genre ML Pipeline API")
loaded_model = None

class ScrapeRequest(BaseModel):
    api_key: str
    start_appid: int = 0
    limit: int = 500


class DatasetRequest(BaseModel):
    steam_csv: str = "steam_games_dataset.csv"
    img_dir: str = "dataset_images"
    output_csv: str = "dataset.csv"


class TrainRequest(BaseModel):
    csv_path: str = "data/dataset.csv"
    img_dir: str = "data/dataset_images"
    output_dir: str = "."
    batch_size: int = 32
    num_workers: int = 4
    num_epochs_frozen: int = 5
    num_epochs_unfrozen: int = 10

@app.post("/scrape")
def scrape_games(req: ScrapeRequest):
    try:
        scraper = SteamScraper(api_key=req.api_key)
        scraper.run(start_appid=req.start_appid, limit=req.limit)
        return {"status": "scrape_done", "file": scraper.data_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_images")
def download_screens():
    try:
        download_images()
        return {"status": "images_downloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_dataset")
def create_dataset(req: DatasetRequest):
    try:
        df = generate_dataset_csv(
            steam_csv=req.steam_csv,
            img_dir=req.img_dir,
            output_csv=req.output_csv,
        )
        return {
            "status": "dataset_created",
            "rows": len(df),
            "output_csv": req.output_csv,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model(req: TrainRequest):
    try:
        result = run_training(
            csv_path=req.csv_path,
            img_dir=req.img_dir,
            output_dir=req.output_dir,
            batch_size=req.batch_size,
            num_workers=req.num_workers,
            num_epochs_frozen=req.num_epochs_frozen,
            num_epochs_unfrozen=req.num_epochs_unfrozen,
            device=DEVICE,
            verbose=True,
        )
        return {
            "status": "training_finished",
            "best_f1": result["best_f1"],
            "metrics": result["final_metrics"],
            "model_path": result["model_path"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_model")
def load_checkpoint(model_path: str):
    global loaded_model

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Checkpoint introuvable")

    try:
        loaded_model = load_model(model_path, device=DEVICE)
        return {"status": "model_loaded", "path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

DEFAULT_CHECKPOINT = "best_model_final.pth"

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), checkpoint_path: Optional[str] = None):
    global loaded_model

    # Si aucun modèle chargé, utiliser checkpoint fourni ou default
    if loaded_model is None:
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint introuvable: {checkpoint_path}")
        try:
            loaded_model = load_model(checkpoint_path, device=DEVICE)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur chargement modèle: {e}")

    # sauvegarde temporaire
    os.makedirs("temp", exist_ok=True)
    temp_name = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join("temp", temp_name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # prétraitement et prédiction
        tensor = preprocess_image(temp_path)
        results = predict(loaded_model, tensor)
    finally:
        # cleanup même en cas d'erreur
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"results": results}
