"""Central configuration for the RECALL topic classifier."""
from pathlib import Path
import os
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR.parent / ".env")

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"

# Hugging Face model hosting configuration
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "farhanx3rz/memorymind-classifier").strip()
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
def resolve_model_dir(explicit_path: Path | str | None = None) -> Path:
    """Robustly resolve the classifier checkpoint directory across project layouts.

    Resolution precedence:
    1. Explicit function argument if given.
    2. RECALL_MODEL_DIR or RECALL_CLASSIFIER_PATH environment variables.
    3. backend/ml/models/recall-topic-classifier
    4. workspace_root/models/recall-topic-classifier
    5. backend/models/recall-topic-classifier
    """
    if explicit_path is not None:
        return Path(explicit_path).resolve()

    candidates: list[Path] = []
    env_dir = os.getenv("RECALL_MODEL_DIR") or os.getenv("RECALL_CLASSIFIER_PATH")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend([
        ML_DIR / "models" / "recall-topic-classifier",
        BACKEND_DIR.parent / "models" / "recall-topic-classifier",
        BACKEND_DIR / "models" / "recall-topic-classifier",
    ])

    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            has_weights = (
                (candidate / "model.safetensors").exists()
                or (candidate / "pytorch_model.bin").exists()
            )
            if has_weights:
                return candidate.resolve()

    return (ML_DIR / "models" / "recall-topic-classifier").resolve()


MODEL_DIR = resolve_model_dir()

LABELS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Large Language Models",
    "RAG",
    "Information Retrieval",
    "Data Science",
    "Programming",
    "Web Development",
    "Databases",
    "Cloud Computing",
    "Cybersecurity",
    "Mathematics",
    "Science",
    "Business",
    "Other",
    "History",
]

MODEL_NAME = os.getenv("RECALL_CLASSIFIER_MODEL", "distilbert-base-uncased")
LEARNING_RATE = float(os.getenv("RECALL_CLASSIFIER_LR", "2e-5"))
BATCH_SIZE = int(os.getenv("RECALL_CLASSIFIER_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("RECALL_CLASSIFIER_EPOCHS", "3"))
MAX_LENGTH = int(os.getenv("RECALL_CLASSIFIER_MAX_LENGTH", "256"))
CHUNK_SIZE = int(os.getenv("RECALL_CLASSIFIER_CHUNK_SIZE", "220"))
CHUNK_OVERLAP = int(os.getenv("RECALL_CLASSIFIER_CHUNK_OVERLAP", "40"))
THRESHOLD = float(os.getenv("RECALL_CLASSIFIER_THRESHOLD", "0.5"))
SEED = int(os.getenv("RECALL_CLASSIFIER_SEED", "42"))
TOP_K_CHUNKS = int(os.getenv("RECALL_CLASSIFIER_TOP_K_CHUNKS", "3"))

TRAIN_FILE = DATA_DIR / "starter_dataset.jsonl"
SPLITS_FILE = DATA_DIR / "splits.json"
