from typing import List, Literal, Dict, Any
from pydantic import BaseModel

class FitRequest(BaseModel):
    hyperparametes: Dict[str, Any]

class PredictRequest(BaseModel):
    X: List[List[float]]

class GetStatusResponse(BaseModel):
    status: str
    models: List[str]

class MessageResponse(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    predictions: List[float]

class FitResponse(BaseModel):
    message: str

class ModelListResponse(BaseModel):
    models: List[Dict[str, str]]

class ProcessStatusResponse(BaseModel):
    status: str
    processes: List[Dict[str, str]]

class GetStatusResponse(BaseModel):
    status: str
    models: List[str]

class DeleteResponse(BaseModel):
    message: str

class CollisionResolver:
    mode: Literal["min", "max"]