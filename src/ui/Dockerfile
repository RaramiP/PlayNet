FROM python:3.12-slim

WORKDIR /ui

COPY src/ui/ .

# Installer les dépendances
RUN pip install --no-cache-dir streamlit requests pillow dotenv

EXPOSE 8501

CMD ["streamlit", "run", "simple_ui.py", "--server.address=0.0.0.0"]
