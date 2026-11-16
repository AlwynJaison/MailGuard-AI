"""Central configuration for MailGuard AI."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
DATABASE_DIR = BASE_DIR / "database"
MODEL_PATH = MODEL_DIR / "spam_classifier.keras"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "evaluation.json"
PLOT_PATH = MODEL_DIR / "training_history.png"
DATABASE_PATH = DATABASE_DIR / "history.db"
MAX_MESSAGE_LENGTH = 20_000
MAX_FEATURES = 20_000
RANDOM_STATE = 42
