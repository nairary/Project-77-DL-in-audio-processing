import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from api import v1_router

from settings.config import MODELS_DIR

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )

@app.get("/")
async def root():
    return {"message": "Hello, this is my API service"}


## Реализуйте роутер с префиксом /api/v1/models
app.include_router(v1_router, prefix="/api/v1/models")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)