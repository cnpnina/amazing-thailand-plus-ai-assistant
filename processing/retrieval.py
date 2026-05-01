from storage.qdrant_store import COLLECTION, get_qdrant_client
from processing.rerank import mmr_rerank
from processing.embedder import embed_texts
import re
from pythainlp.tokenize import word_tokenize

STOPWORDS = {
    "อยาก", "ช่วย", "หน่อย", "ตามรอย", "วางแพลน", "series", "ซีรีย์", "ซีรีส์",
    "trip", "travel", "plan", "footsteps", "can", "you", "help", "me", "the", "of",
}


def _keywords_from_query(query: str):
    base_tokens = re.findall(r"[\u0E00-\u0E7Fa-zA-Z0-9\u4E00-\u9FFF]+", query.lower())
    thai_tokens = []
    try:
        thai_tokens = [
            token.strip()
            for token in word_tokenize(query.lower(), engine="newmm", keep_whitespace=False)
            if token.strip()
        ]
    except Exception:
        thai_tokens = []

    tokens = base_tokens + thai_tokens
    keywords = [token for token in tokens if len(token) >= 2 and token not in STOPWORDS]
    return list(dict.fromkeys(keywords))


def _keyword_overlap_score(item: dict, keywords):
    haystack = f"{item.get('title', '')} {item.get('text', '')}".lower()
    return sum(1 for keyword in keywords if keyword in haystack)

def embed_query(query: str):
    return embed_texts([query], task="retrieval")[0]


def retrieve_candidates(query: str, limit=12):
    client = get_qdrant_client()
    query_vector = embed_query(query)
    try:
        collection_info = client.get_collection(collection_name=COLLECTION)
        vector_cfg = collection_info.config.params.vectors
        if hasattr(vector_cfg, "size"):
            expected_size = int(vector_cfg.size)
        elif isinstance(vector_cfg, dict):
            first_cfg = next(iter(vector_cfg.values()))
            expected_size = int(first_cfg.size)
        else:
            expected_size = len(query_vector)
    except Exception:
        expected_size = len(query_vector)

    if len(query_vector) != expected_size:
        # Avoid runtime crash on mixed-dimension indexes; caller will handle empty evidence.
        return query_vector, []

    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=limit,
            with_vectors=True,
            with_payload=True,
        )
    except ValueError:
        # Common case: stale collection with different vector size.
        return query_vector, []
    results = getattr(response, "points", response)

    candidates = []
    for point in results:
        payload = point.payload or {}
        text = payload.get("text", "")
        vector = point.vector
        if isinstance(vector, dict):
            vector = vector.get("", []) or next(iter(vector.values()), [])
        if not text:
            continue
        if not vector:
            continue
        candidates.append(
            {
                "id": str(point.id),
                "score": float(point.score),
                "vector": vector,
                "text": text,
                "source_url": payload.get("source_url"),
                "title": payload.get("title"),
            }
        )

    return query_vector, candidates


def retrieve(query: str, candidate_limit=12, top_k=6):
    query_vector, candidates = retrieve_candidates(query=query, limit=candidate_limit)
    reranked = mmr_rerank(query_vector=query_vector, candidates=candidates, top_k=top_k)
    keywords = _keywords_from_query(query)
    min_keep = min(6, len(reranked))
    if keywords:
        strict_matches = [item for item in reranked if _keyword_overlap_score(item, keywords) > 0]
        if strict_matches:
            # Keep relevant matches first, then backfill to maintain enough evidence context.
            merged = []
            seen_ids = set()
            for item in strict_matches + reranked:
                item_id = item.get("id")
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                merged.append(item)
                if len(merged) >= min_keep:
                    break
            reranked = merged
        else:
            # Avoid empty-evidence regressions on Thai/Chinese tokenization edge cases.
            reranked = reranked[:min_keep]
    for item in reranked:
        item.pop("vector", None)
    return reranked