import hashlib
import logging
import os
import re
import warnings

from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="Default prompt name is set to 'document'.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Keep fallback dimensionality aligned with jina-embeddings-v5 output.
EMBED_DIM = 1024
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"
_model = None
_use_local_fallback = False


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _hash_embedding(text: str):
    vec = [0.0] * EMBED_DIM
    tokens = _tokenize(text)
    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % EMBED_DIM
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign

    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _get_model():
    global _model, _use_local_fallback

    if _use_local_fallback:
        return None
    if _model is not None:
        return _model

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        _model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception:
        _use_local_fallback = True
        _model = None
    return _model


def embed_texts(chunks, task="retrieval"):
    model = _get_model()
    if model is None:
        return [_hash_embedding(chunk) for chunk in chunks]

    safe_chunks = [chunk[:1000] for chunk in chunks]
    try:
        embeddings = model.encode(
            safe_chunks,
            normalize_embeddings=True,
            task=task,
            batch_size=4,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except RuntimeError:
        # Runtime OOM or oversized attention buffer: fallback to local embeddings.
        return [_hash_embedding(chunk) for chunk in safe_chunks]

    return embeddings