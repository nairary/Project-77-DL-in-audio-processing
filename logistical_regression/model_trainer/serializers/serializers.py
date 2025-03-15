from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field

# class Hyperparameters(BaseModel):
#     # Ограничения на гиперпараметры
#     n_jobs: Optional[int] = -1
#     penalty: Optional[Literal["l2", "l1", "elasticnet"]] = "l2"
#     loss: Optional[
#         Literal[
#             "hinge", "log_loss", "log", "modified_huber", "squared_hinge",
#             "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
#         ]
#     ] = "log_loss"
#     optional: Optional[Dict[str, Any]] = None

# class FitRequest(BaseModel):
#     id: int
#     hyperparameters: Hyperparameters

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

class GetStatusResponse(BaseModel):
    status: str
    processes: List[str]

class FitRequest(BaseModel):
    id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

class ModelEvaluationRequest(BaseModel):
    model_name: str
    test_split: float = Field(default=0.2, ge=0.0, le=1.0)
    random_state: int = Field(default=42)

class ModelEvaluationResponse(BaseModel):
    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    one_semitone_accuracy: float
    silence_accuracy: float
    non_silence_accuracy: float
    num_test_samples: int
    results_path: str

class HyperparameterTuningRequest(BaseModel):
    model_name: str
    parameter_grid: Dict[str, List[Any]] = Field(
        default_factory=lambda: {
            "loss": ["log_loss", "hinge"],
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01, 0.1]
        }
    )
    cv_folds: int = Field(default=3, ge=2, le=10)
    scoring_metric: str = Field(default="f1_weighted")
    n_jobs: int = Field(default=-1)
    max_samples: int = Field(default=10000, description="Maximum samples to use for tuning, set to 0 to use all")
    save_best_model: bool = Field(default=True)

class HyperparameterTuningResponse(BaseModel):
    best_params: Dict[str, Any]
    best_score: float
    model_path: Optional[str] = None
    results_path: str