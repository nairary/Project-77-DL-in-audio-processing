from fastapi import APIRouter, HTTPException, UploadFile, File
from http import HTTPStatus
from typing import List, Dict

from services.model_manager import (
    extract_and_save_data,
    train_model,
    predict_model,
    get_models,
)
from services.process_manager import (
    start_process,
    end_process,
)
from serializers.serializers import (
    FitRequest,
    CollisionResolver,
    MessageResponse,
    PredictionResponse,
    FitResponse,
    ModelListResponse,
)

router = APIRouter()

@router.post("/extract_features", response_model=MessageResponse, status_code=HTTPStatus.OK)
async def extract_features(resolver: CollisionResolver):
    process_id = "extract_features"
    try:
        await start_process(process_id, "extract_features")
        extract_and_save_data(resolver.mode)
        await end_process(process_id)
        return MessageResponse(message="Audio and MIDI features were extracted successfully")
    except Exception as e:
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/fit", response_model=List[FitResponse], status_code=HTTPStatus.OK)
async def fit(request: FitRequest):
    responses = []
    process_id = f"fit_{request.hyperparametes}"
    try:
        await start_process(process_id, "fit")
        train_model(request.hyperparametes)
        responses.append(FitResponse(message=f"Model {request.id} saved successfully"))
        await end_process(process_id)
    except Exception as e:
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))
    return responses

@router.post("/predict", response_model=PredictionResponse, status_code=HTTPStatus.OK)
async def predict(file: UploadFile = File(...)):
    process_id = "predict"
    try:
        await start_process(process_id, "predict")
        predictions = predict_model(file)
        await end_process(process_id)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/list_models", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def list_models():
    try:
        models = get_models()
        return ModelListResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))