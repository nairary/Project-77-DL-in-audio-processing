import os

BASE_DIR = "/logistical_regression"

# upload_data
MP3_VOCALS_DIR = os.path.join(BASE_DIR, "data", "mp3_vocals")
MIDI_VOCALS_DIR = os.path.join(BASE_DIR, "data", "midi_vocals")
MATCH_SCORES_PATH = os.path.join(BASE_DIR, "data", "match_scores.json")

# train_model
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "data", "models", "default_model.pkl") 
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")
FEATURES_DIR = UPLOAD_DIR = os.path.join(BASE_DIR, "data", "features")

# predict_model
MODEL_NAME = None

# misc
MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 4))
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 2))