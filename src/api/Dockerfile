FROM python:3.14.2
WORKDIR /app

RUN pip install fastapi pydantic uvicorn requests pandas numpy torch torchvision pillow tqdm

COPY ./ ./ 
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
