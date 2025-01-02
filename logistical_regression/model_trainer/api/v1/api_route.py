import io
import sys, os
from fastapi import APIRouter, HTTPException, UploadFile, File
from http import HTTPStatus
from typing import List, Dict, Optional

from services.model_manager import (
    extract_and_save_data,
    fit,
    predict_model,
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
)

from settings.config import (MP3_VOCALS_DIR, MIDI_VOCALS_DIR, MATCH_SCORES_PATH, MODELS_DIR, FEATURES_DIR, PREDICTIONS_DIR)

router = APIRouter()

UPLOAD_DIR = FEATURES_DIR

@router.post("/upload_data")
async def extract_features(
    file: Optional[UploadFile] = File(None),
    payload: Optional[Dict[str, str]] = None
):
    """
    Формирует датасет из загруженного файла .npz или JSON с путями.
    """
    if file:
        # Обработка загруженного .npz файла
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"message": f"Файл {file.filename} успешно обработан", "path": file_path}

    elif payload:
        # Обработка JSON с путями
        mp3_vocals_root = payload.get("mp3_vocals_root")
        lmd_aligned_vocals_root = payload.get("lmd_aligned_vocals_root")
        match_scores_json = payload.get("match_scores_json")
        output_npz = payload.get("output_npz")

        if not all([mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, output_npz]):
            raise HTTPException(status_code=400, detail="Не все пути указаны в JSON")

        extract_and_save_data(mp3_vocals_root, lmd_aligned_vocals_root, match_scores_json, output_npz)
        return {"message": "Фичи успешно извлечены", "paths": payload}

    else:
        raise HTTPException(status_code=400, detail="Необходимо предоставить файл .npz или JSON с путями")

@router.post("/train_model", response_model=List[FitResponse], status_code=HTTPStatus.OK)
async def train_model(request: FitRequest):
    responses = []
    process_id = f"fit_{request.hyperparametes}"
    try:
        await start_process(process_id, "fit")
        fit(request.hyperparametes)
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
        output = io.StringIO()
        sys.stdout = output
        await start_process(process_id, "predict")
        predictions = predict_model(file)
        await end_process(process_id)
        sys.stdout = sys.__stdout__
        logs = output.getvalue()

        return PredictionResponse(message=logs)
    except Exception as e:
        sys.stdout = sys.__stdout__
        await end_process(process_id)
        raise HTTPException(status_code=400, detail=str(e))