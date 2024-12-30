from .api import v1_router as api_router

from .settings import MODELS_DIR, MAX_PROCESSES, MAX_LOADED_MODELS

from .services import (
    train_model,
    load_model,
    unload_model,
    list_models,
    remove_model,
    remove_all_models,
    predict,
)

__all__ = [
    "api_router",
    "MODELS_DIR",
    "MAX_PROCESSES",
    "MAX_LOADED_MODELS",
    "train_model",
    "load_model",
    "unload_model",
    "list_models",
    "remove_model",
    "remove_all_models",
    "predict",
]