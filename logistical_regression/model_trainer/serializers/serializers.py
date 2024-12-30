from typing import List, Union, Dict, Any
from pydantic import BaseModel

class ModelConfig(BaseModel):
    id: str
    hyperparameters: Dict[str, Any]

class FitRequest(BaseModel):
    hyperparametes: Dict[str, Any]

class LoadRequest(BaseModel):
    id: str

class LoadResponse(BaseModel):
   id: str

class UnloadResponse(BaseModel):
    message: str

class PredictRequest(BaseModel):
    X: List[List[float]]

class DeleteResponse(BaseModel):
    message: str

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