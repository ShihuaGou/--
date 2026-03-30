from pathlib import Path

MODEL_ID = "tiiuae/falcon-7b-instruct"
DEVICE = "cuda:0"
MEMORY_DIM = 384
SHORT_MEMORY_LIMIT = 200
HOT_MEMORY_LIMIT = 1000
BASE_DIR = Path(__file__).resolve().parent
MEMORY_DB_PATH = BASE_DIR / "memory_store"
