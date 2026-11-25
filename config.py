"""Central settings. Values can be overridden through .env or environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DOCUMENTS_DIR = ROOT / "data" / "documents"
UPLOADS_DIR = ROOT / "data" / "uploads"
STORAGE_DIR = ROOT / "storage"
MODEL_CACHE = ROOT / ".cache" / "models"

# Multilingual model (Arabic + English), quantized ONNX, runs on CPU.
EMBED_REPO = "Xenova/multilingual-e5-small"
EMBED_FILE = "onnx/model_quantized.onnx"
EMBED_DIM = 384

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

# Groq free tier allows about 8K tokens/minute per model, so requests stay small.
MAX_AGENT_STEPS = 7
MAX_PROMPT_TOKENS = 5000
MAX_COMPLETION_TOKENS = 1800


def _models(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


AGENT_MODELS = _models("AGENT_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b")
FAST_MODELS = _models("FAST_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b,openai/gpt-oss-120b")


def groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()
