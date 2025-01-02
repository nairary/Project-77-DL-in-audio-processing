import os

# upload_data
MP3_VOCALS_DIR = None
MIDI_VOCALS_DIR = None
MATCH_SCORES_PATH = None

# train_model
MODELS_DIR = os.path.join("logistical_regression", "data", "models")
PREDICTIONS_DIR = None
FEATURES_DIR = UPLOAD_DIR = os.path.join("logistical_regression", "data", "features")

# predict_model
MODEL_NAME = None

# misc
MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 4))
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 2))