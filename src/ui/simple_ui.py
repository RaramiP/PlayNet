import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Steam Game Genre Classifier",
    layout="wide",
)

st.title("Steam Game Genre - ML Pipeline")

# SIDEBAR – PIPELINE
st.sidebar.header("Pipeline automatisé")

# -------- SCRAPING --------
st.sidebar.subheader("1. Scraping Steam")

api_key = st.sidebar.text_input("Steam API Key", type="password", value=API_KEY)
start_appid = st.sidebar.number_input("Start AppID", min_value=0, value=0)
limit = st.sidebar.number_input("Limit", min_value=1, value=100)

if st.sidebar.button("Lancer le scraping"):
    with st.spinner("Scraping en cours..."):
        res = requests.post(
            f"{API_URL}/scrape",
            json={
                "api_key": api_key,
                "start_appid": start_appid,
                "limit": limit,
            },
        )
    st.sidebar.success(res.json())

# -------- DOWNLOAD IMAGES --------
st.sidebar.subheader("2. Télécharger les images")

if st.sidebar.button("Télécharger les images"):
    with st.spinner("Téléchargement des images..."):
        res = requests.post(f"{API_URL}/download_images")
    st.sidebar.success(res.json())

# -------- DATASET --------
st.sidebar.subheader("3. Générer le dataset")

steam_csv = st.sidebar.text_input("Steam CSV", "data/steam_games_dataset.csv")
img_dir = st.sidebar.text_input("Image directory", "data/dataset_images")
output_csv = st.sidebar.text_input("Output CSV", "data/dataset.csv")

if st.sidebar.button("Générer dataset"):
    with st.spinner("Création du dataset..."):
        res = requests.post(
            f"{API_URL}/generate_dataset",
            json={
                "steam_csv": steam_csv,
                "img_dir": img_dir,
                "output_csv": output_csv,
            },
        )
    st.sidebar.success(res.json())

# -------- TRAINING --------
st.sidebar.subheader("4. Entraîner le modèle")

batch_size = st.sidebar.number_input("Batch size", value=32)
epochs_frozen = st.sidebar.number_input("Epochs (frozen)", value=5)
epochs_unfrozen = st.sidebar.number_input("Epochs (unfrozen)", value=10)

if st.sidebar.button("Lancer l'entraînement"):
    with st.spinner("Entraînement en cours (ça peut prendre du temps)..."):
        res = requests.post(
            f"{API_URL}/train",
            json={
                "csv_path": output_csv,
                "img_dir": img_dir,
                "batch_size": batch_size,
                "num_epochs_frozen": epochs_frozen,
                "num_epochs_unfrozen": epochs_unfrozen,
            },
        )
    st.sidebar.success("Entraînement terminé")
    st.sidebar.json(res.json())

# -------- LOAD MODEL --------
st.sidebar.subheader("5. Charger un modèle")

if st.sidebar.button("Charger le modèle"):
    with st.spinner("Chargement du modèle..."):
        res = requests.post(
            f"{API_URL}/load_model",
            params={"model_path": "best_model_final.pth"},
        )
    st.sidebar.success(res.json())


# MAIN
st.header("Prédiction sur une image")

uploaded_file = st.file_uploader(
    "Upload une image de jeu (screenshot, cover, etc.)",
    type=["jpg", "jpeg", "png"],
)

# Chemin par défaut du checkpoint
checkpoint_default = "best_model_final.pth"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image envoyée", width=min(image.width, 400))


    if st.button("Prédire le genre"):
        with st.spinner("Prédiction en cours..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            # Récupère le checkpoint fourni, sinon utilise le default
            checkpoint_path = st.text_input(
                "Checkpoint (optionnel si déjà chargé)", ""
            ).strip()
            if checkpoint_path == "":
                checkpoint_path = checkpoint_default

            params = {"checkpoint_path": checkpoint_path}

            try:
                res = requests.post(f"{API_URL}/predict", files=files, params=params)
                res.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur API : {e}")
            else:
                results = res.json()["results"]
                st.subheader("Résultats")
                for r in results:
                    st.write(f"**{r['genre']}** : {r['probability']:.2f}")
                    st.progress(r["probability"])
