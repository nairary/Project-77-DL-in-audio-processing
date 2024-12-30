from fastapi import APIRouter, HTTPException, UploadFile, File
from http import HTTPStatus
from typing import List, Dict

from services.model_manager import (
    extract_and_save_data,
    train_model,
    predict_model,
    unload_model,
    get_models,
    remove_model,
    remove_all
)
from services.process_manager import (
    start_process,
    end_process,
    get_active_processes
)
from serializers.serializers import (
    FitRequest,
    ModelConfig,
    LoadRequest,
    LoadResponse,
    UnloadResponse,
    PredictRequest,
    DeleteResponse,
    GetStatusResponse,
    PredictRequest,
    MessageResponse,
    PredictionResponse,
    FitResponse,
    ModelListResponse,
    ProcessStatusResponse,
    GetStatusResponse,
    DeleteResponse
)

router = APIRouter()

@router.post("/extract_features", response_model=MessageResponse, status_code=HTTPStatus.OK)
async def extract_features():
    process_id = "extract_features"
    try:
        await start_process(process_id, "extract_features")
        extract_and_save_data()
        await end_process(process_id)
        return MessageResponse(message="Audio and MIDI features were extracted successfully")
    except Exception as e:
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/fit", response_model=List[FitResponse], status_code=HTTPStatus.OK)
async def fit(requests: List[FitRequest]):
    responses = []
    for request in requests:
        process_id = f"fit_{request.config.id}"
        try:
            await start_process(process_id, "fit")
            train_model(request.config.id, request.X, request.y, request.config.dict())
            responses.append(FitResponse(message=f"Model {request.config.id} saved successfully"))
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

@router.post("/unload", response_model=MessageResponse, status_code=HTTPStatus.OK)
async def unload():
    try:
        unload_model()
        return MessageResponse(message="Inference space cleared")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/list_models", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def list_models():
    try:
        models = get_models()
        return ModelListResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get_status", response_model=ProcessStatusResponse, status_code=HTTPStatus.OK)
async def get_status():
    try:
        active_processes = get_active_processes()
        if not active_processes:
            return ProcessStatusResponse(status="No active processes", processes=[])

        processes = [{"id": pid, "type": ptype} for pid, ptype in active_processes.items()]
        return ProcessStatusResponse(status="Active processes", processes=processes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/remove/{model_id}", response_model=DeleteResponse, status_code=HTTPStatus.OK)
async def remove(model_id: str):
    try:
        remove_model(model_id)
        return DeleteResponse(message=f"Model {model_id} removed successfully")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/remove_all", response_model=DeleteResponse, status_code=HTTPStatus.OK)
async def remove_all_models():
    try:
        remove_all()
        return DeleteResponse(message="All models removed successfully")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
