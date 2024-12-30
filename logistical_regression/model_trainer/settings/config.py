from decouple import config

MIDI_DIR = config("MIDI_VOCALS_DIR", default="data/lmd_aligned")
MP3_VOCALS_DIR = config("MP3_VOCALS_DIR", default="data/mp3_vocals")
MIDI_VOCALS_DIR = config("MIDI_VOCALS_DIR", default="data/lmd_aligned_vocals")
MATCH_SCORES_PATH = config("MATCH_SCORES_PATH", default="data/match-scores.json")

MODELS_DIR = config("MODELS_DIR", default="data/models/")
FEATURES_DIR = config("FEATURES_DIR", default="data/feats/")

MAX_PROCESSES = config("MAX_PROCESSES", default=4, cast=int)
MAX_LOADED_MODELS = config("MAX_LOADED_MODELS", default=2, cast=int)