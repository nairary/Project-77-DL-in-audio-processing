import os

MIDI_DIR = os.getenv("MIDI_VOCALS_DIR", "data/lmd_aligned")
MP3_VOCALS_DIR = os.getenv("MP3_VOCALS_DIR", "data/mp3_vocals")
MIDI_VOCALS_DIR = os.getenv("MIDI_VOCALS_DIR", "data/lmd_aligned_vocals")
MATCH_SCORES_PATH = os.getenv("MATCH_SCORES_PATH", "data/match-scores.json")

MODELS_DIR = os.getenv("MODELS_DIR", "data/models/")
PREDICTIONS_DIR = os.getenv("PREDICTIONS_DIR", "data/predictions/")
FEATURES_DIR = os.getenv("FEATURES_DIR", "data/feats/")

MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 4))
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 2))