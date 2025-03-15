import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier
from typing import Dict, Any, List, Optional
import os
import joblib
import json
import asyncio

from settings.config import MODELS_DIR, FEATURES_DIR, PREDICTIONS_DIR
from serializers.serializers import (ModelEvaluationResponse, HyperparameterTuningRequest, HyperparameterTuningResponse)

################################ HYPERPARAMETER TUNING ################################

async def tune_hyperparameters(request: HyperparameterTuningRequest):
    """
    Performs GridSearchCV to find optimal hyperparameters for the SGDClassifier.
    """
    # Run the CPU-intensive task in a separate thread to not block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_hyperparameter_tuning, request)
    return result

def _run_hyperparameter_tuning(request: HyperparameterTuningRequest):
    """Non-async version to run in executor"""
    # Load dataset
    npz_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.npz')]
    if not npz_files:
        raise Exception("No dataset found in features directory")
    
    first_npz_file = os.path.join(FEATURES_DIR, npz_files[0])
    data = np.load(first_npz_file)
    X = data["X"]
    y = data["Y"]
    
    print(f"[INFO] Loaded dataset for hyperparameter tuning, X={X.shape}, y={y.shape}")
    
    # If dataset is too large, sample a portion for faster tuning
    if X.shape[0] > request.max_samples and request.max_samples > 0:
        indices = np.random.choice(X.shape[0], request.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"[INFO] Sampled {request.max_samples} examples for faster tuning")
    
    # Get parameter grid directly - no need to call .dict()
    # If request.parameter_grid is already a dict, use it directly
    param_grid = request.parameter_grid
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=request.cv_folds, shuffle=True, random_state=42)
    
    # Set up classifier with warm_start for partial fitting
    clf = SGDClassifier(warm_start=True, max_iter=100, tol=1e-3)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv,
        scoring=request.scoring_metric,
        n_jobs=request.n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    # Fit the grid search
    print(f"[INFO] Starting GridSearchCV with {len(list(cv.split(X, y)))} folds")
    grid_search.fit(X, y)
    
    # Get results
    cv_results = grid_search.cv_results_
    
    # Save best model if requested
    model_path = None
    if request.save_best_model:
        best_model = grid_search.best_estimator_
        model_path = os.path.join(MODELS_DIR, f"{request.model_name}_best.pkl")
        joblib.dump(best_model, model_path)
        print(f"[INFO] Saved best model to {model_path}")
    
    # Prepare results for JSON serialization
    serializable_results = {
        "best_params": {},
        "cv_results": {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
            "rank_test_score": []
        }
    }
    
    # Convert best_params to serializable format
    for k, v in grid_search.best_params_.items():
        if v is None:
            serializable_results["best_params"][k] = None
        elif isinstance(v, (bool, int, float, str)):
            serializable_results["best_params"][k] = v
        else:
            serializable_results["best_params"][k] = str(v)
    
    # Convert cv_results to serializable format
    for i, params in enumerate(cv_results["params"]):
        param_dict = {}
        for k, v in params.items():
            if v is None:
                param_dict[k] = None
            elif isinstance(v, (bool, int, float, str)):
                param_dict[k] = v
            else:
                param_dict[k] = str(v)
                serializable_results["cv_results"]["params"].append(param_dict)
        serializable_results["cv_results"]["mean_test_score"].append(float(cv_results["mean_test_score"][i]))
        serializable_results["cv_results"]["std_test_score"].append(float(cv_results["std_test_score"][i]))
        serializable_results["cv_results"]["rank_test_score"].append(int(cv_results["rank_test_score"][i]))
    
    # Save results to JSON file
    results_path = os.path.join(MODELS_DIR, f"{request.model_name}_tuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return HyperparameterTuningResponse(
        best_params=serializable_results["best_params"],
        best_score=float(grid_search.best_score_),
        model_path=model_path,
        results_path=results_path
    )

################################ MODEL EVALUATION ################################

async def evaluate_model(model_name: str, test_split: float = 0.2, random_state: int = 42):
    """
    Evaluates a trained model on a test set and returns performance metrics.
    
    Parameters:
    model_name: Name of the model to evaluate
    test_split: Fraction of data to use for testing
    random_state: Random seed for reproducibility
    
    Returns:
    Dictionary with evaluation metrics
    """
    # Load model
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model from {model_path}")
    except FileNotFoundError:
        return {"error": f"Model {model_name} not found"}
    
    # Load dataset
    npz_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.npz')]
    if not npz_files:
        return {"error": "No dataset found in features directory"}
    
    first_npz_file = os.path.join(FEATURES_DIR, npz_files[0])
    data = np.load(first_npz_file)
    X = data["X"]
    y = data["Y"]
    
    # Split into train/test
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0] * test_split)
    test_indices = indices[:test_size]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"[INFO] Evaluating on test set with {X_test.shape[0]} examples")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate macro and weighted F1 scores
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate class distribution in test set
    unique_classes, class_counts = np.unique(y_test, return_counts=True)
    class_distribution = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
    
    # Calculate silence vs non-silence accuracy
    silence_indices = (y_test == 0)
    non_silence_indices = (y_test != 0)
    
    silence_accuracy = accuracy_score(y_test[silence_indices], y_pred[silence_indices]) if any(silence_indices) else 0
    non_silence_accuracy = accuracy_score(y_test[non_silence_indices], y_pred[non_silence_indices]) if any(non_silence_indices) else 0
    
    # Calculate one_semitone_accuracy (±1 accuracy for non-silence classes)
    # Filter only the positive classes (non-silence, assuming 0 is silence)
    positive_indices = (y_test > 0)
    
    if any(positive_indices):
        # Get actual and predicted values for positive classes
        y_test_positive = y_test[positive_indices]
        y_pred_positive = y_pred[positive_indices]
        
        # Calculate accuracy with ±1 tolerance
        semitone_correct = np.abs(y_test_positive - y_pred_positive) <= 1
        one_semitone_accuracy = np.mean(semitone_correct)
    else:
        one_semitone_accuracy = 0
    
    # Prepare evaluation results
    evaluation_results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "one_semitone_accuracy": float(one_semitone_accuracy),
        "class_distribution": class_distribution,
        "silence_accuracy": float(silence_accuracy),
        "non_silence_accuracy": float(non_silence_accuracy),
        "classification_report": report
    }
    
    # Save evaluation results
    results_path = os.path.join(MODELS_DIR, f"{model_name}_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return ModelEvaluationResponse(
        model_name=model_name,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        one_semitone_accuracy=float(one_semitone_accuracy),
        silence_accuracy=float(silence_accuracy),
        non_silence_accuracy=float(non_silence_accuracy),
        num_test_samples=int(X_test.shape[0]),
        results_path=results_path
    )