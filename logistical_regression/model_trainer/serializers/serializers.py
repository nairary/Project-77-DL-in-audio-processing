from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel

class FitRequest(BaseModel):
    id: int
    hyperparameters: Dict[str, Any]

    # Ограничения на гиперпараметры
    n_jobs: Optional[int] = None
    penalty: Optional[Literal["l2", "l1", "elasticnet"]] = None
    loss: Optional[
        Literal[
            "hinge", "log_loss", "log", "modified_huber", "squared_hinge",
            "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
        ]
    ] = None
    optional: Optional[Dict[str, Any]] = None

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