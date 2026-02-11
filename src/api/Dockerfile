FROM python:3.12-slim

WORKDIR /api

COPY src/api/ .
COPY src/utils/ ./utils
COPY src/model/ ./model
COPY data/ ./data
COPY best_model_final.pth ./best_model_final.pth
COPY best_model_phase1.pth ./best_model_phase1.pth

# Installer dépendances
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] requests pandas numpy torch torchvision pillow tqdm pydantic python-multipart

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
