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

class MessageResponse(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    message: str
    midi_notes: List[int] 

class FitResponse(BaseModel):
    message: str

class ModelListResponse(BaseModel):
    message: List[str]

class CollisionResolver(BaseModel):
    mode: Literal["min", "max"]

class ModelNameRequest(BaseModel):
    model_name: str