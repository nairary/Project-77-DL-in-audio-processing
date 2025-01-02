import asyncio
import sys, os
import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from http import HTTPStatus
from typing import List, Dict, Optional

from services.model_manager import (
    run_extract_and_save_data,
    fit,
    predict_model
)
from services.process_manager import (
    start_process,
    end_process,
    get_active_processes
)
from serializers.serializers import (
    FitRequest,
    ModelNameRequest,
    ModelListResponse,
    PredictionResponse,
    FitResponse,
    GetStatusResponse
)

from settings.config import (MODELS_DIR, FEATURES_DIR, MODEL_NAME)

router = APIRouter()

UPLOAD_DIR = FEATURES_DIR

@router.post("/upload_data")
async def extract_features(
    file: Optional[UploadFile] = File(None),
    payload: Optional[str] = Form(None)
):
    """
    Формирует датасет из загруженного файла .npz или JSON с путями.
    пример JSON пэйлоада:
    {
    "mp3_vocals_root": "C:/ds/test_upload/mp3_vocals",
    "lmd_aligned_vocals_root": "C:/ds/test_upload/lmd_aligned_vocals",
    "match_scores_json": "C:/ds/test_upload/match-scores.json",
    "output_npz": "D:/Project-77-DL-in-audio-processing/logistical_regression/data/features/my_npz"
}
    """
    if file:
        # Обработка загруженного .npz файла
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка записи файла: {str(e)}")
        
        return {"message": f"Файл {file.filename} успешно обработан", "path": file_path}

    elif payload:

        try:
            payload_dict = json.loads(payload)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Некорректный формат JSON")

        # Обработка JSON с путями
        mp3_vocals_root = payload_dict.get("mp3_vocals_root")
        lmd_aligned_vocals_root = payload_dict.get("lmd_aligned_vocals_root")
        match_scores_json = payload_dict.get("match_scores_json")
        output_npz = payload_dict.get("output_npz")

        if not all([mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, output_npz]):
            raise HTTPException(status_code=400, detail="Не все пути указаны в JSON")
        
        process_id = len(get_active_processes())+1
        try:
            await start_process(process_id, "features extraction")
            await run_extract_and_save_data(mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, output_npz)
            await end_process(process_id)
        except Exception as e:
            await end_process(process_id)
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "Фичи успешно извлечены", "paths": payload_dict}

    else:
        raise HTTPException(status_code=400, detail="Необходимо предоставить файл .npz или JSON с путями")

@router.post("/train_model", response_model=List[FitResponse], status_code=HTTPStatus.OK)
async def train_model(request: FitRequest = Body(...)):
    responses = []

    if not FEATURES_DIR or FEATURES_DIR.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Features folder is not set. Please upload data first."
        )
    
    if not os.path.exists(FEATURES_DIR):
        raise HTTPException(
            status_code=400,
            detail=f"Features folder '{FEATURES_DIR}' does not exist. Please upload data first."
        )
    
    npz_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.npz')]
    if not npz_files:
        raise HTTPException(
            status_code=400,
            detail=f"No .npz files found in features folder '{FEATURES_DIR}'. Please upload data first."
        )

    process_id = f"fit_{request.id}"
    try:
        await start_process(process_id, "fit")
        fit(request)
        responses.append(FitResponse(message=f"Model {request.id} saved successfully"))
        await end_process(process_id)
    except Exception as e:
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))
    return responses

@router.get("/get_model_list", response_model=ModelListResponse, status_code=HTTPStatus.OK)
async def get_model_list():
    """
    Возвращает список всех MODEL_NAME из директории MODEL_DIR.
    """
    if not os.path.exists(MODELS_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"Model directory '{MODELS_DIR}' does not exist."
        )
    
    try:
        model_names = [
            f for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing model directory: {str(e)}"
        )
    
    if not model_names:
        raise HTTPException(
            status_code=404,
            detail=f"No models found in directory '{MODELS_DIR}'."
        )

    return ModelListResponse(message=model_names)

@router.post("/set_model")
async def set_model(request: ModelNameRequest):
    """
    Назначает указанное имя модели в MODEL_NAME.
    """
    global MODEL_NAME
    if not request.model_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Model name cannot be empty or whitespace."
        )
    
    MODEL_NAME = request.model_name.strip()
    return {"message": "Model name set successfully", "model_name": MODEL_NAME}

@router.post("/get_selected_model")
async def get_selected_model():
    """
    Назначает указанное имя модели в MODEL_NAME.
    """
    global MODEL_NAME
    if not MODEL_NAME:
        raise HTTPException(
            status_code=404,
            detail=f"Model {MODEL_NAME} doesn't exist."
        )
    
    return {"message": "Successfully got model name", "model_name": MODEL_NAME}

@router.post("/predict", response_model=PredictionResponse, status_code=HTTPStatus.OK)
async def predict(file: UploadFile = File(...)):
    global MODEL_NAME
    process_id = "predict"
    try:
        await start_process(process_id, "predict")
        predictions = predict_model(MODEL_NAME, file)
        await end_process(process_id)
        return {"message": "Successfully made prediction", "midi_notes": predictions}
    except Exception as e:
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/get_status", response_model=GetStatusResponse)
async def get_status():
    try:
        active_processes = get_active_processes()
        if not active_processes:
            return {"status": "No active models", "models": []}

        models = list(active_processes.keys())
        return {"status": "Active models", "models": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))