import uvicorn
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict
import joblib
from api import v1_router

from settings.config import DEFAULT_MODEL_DIR

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Инициализация Lifespan...")
    try:
        app.state.MODEL_NAME = os.path.basename(DEFAULT_MODEL_DIR)
        print(f"[INFO] Загружена модель: {app.state.MODEL_NAME}")
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке модели: {e}")
        app.state.MODEL_NAME = None
    yield
    print("[INFO] Очистка ресурсов при завершении приложения...")
    app.state.MODEL_NAME = None

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )

@app.get("/")
async def root():
    return {"message": "Hello, this is my API service"}

app.include_router(v1_router, prefix="/api/v1/models")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)