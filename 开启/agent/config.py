from pathlib import Path
import torch
from transformers import BitsAndBytesConfig

LOCAL_MODEL_PATH = Path("/home/goushihua/model/math_shepherd")
MODEL_ID = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "tiiuae/falcon-7b-instruct"  # 优先使用本地模型目录
DEVICE = "cuda:0"
VECTOR_DIM = 384
MEMORY_DIM = VECTOR_DIM
SHORT_MEMORY_LIMIT = 200
HOT_MEMORY_LIMIT = 1000
MAX_SUPER_SHORT = 32
SHORT_MEM_MAX_SIZE = 512 * 1024 * 1024  # 512MB
HOT_MEM_MAX_SIZE = 1024 * 1024 * 1024   # 1GB
ENABLE_COMPRESSION = True
VECTOR_QUANTIZE_TYPE = "INT8"
GLOBAL_DEVICE = "cuda:0"  # A100单卡统一设备，Windows下适配RTX3060
BASE_DIR = Path(__file__).resolve().parent
MEMORY_DB_PATH = BASE_DIR / "memory_store"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 4bit量化配置（A100 80G bf16）
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
