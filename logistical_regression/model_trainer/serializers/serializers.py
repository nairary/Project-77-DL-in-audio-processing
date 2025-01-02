from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel

class Hyperparameters(BaseModel):
    # Ограничения на гиперпараметры
    n_jobs: Optional[int] = -1
    penalty: Optional[Literal["l2", "l1", "elasticnet"]] = "l2"
    loss: Optional[
        Literal[
            "hinge", "log_loss", "log", "modified_huber", "squared_hinge",
            "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
        ]
    ] = "log_loss"
    optional: Optional[Dict[str, Any]] = None

class FitRequest(BaseModel):
    id: int
    hyperparameters: Hyperparameters

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

class CollisionResolver(BaseModel):
    mode: Literal["min", "max"]