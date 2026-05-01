from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import uuid

COLLECTION = "thai_history"
_client = None


def get_qdrant_client():
    global _client
    if _client is None:
        # Local mode for quick testing without Docker.
        _client = QdrantClient(path="./.qdrant_local")
    return _client

def ensure_collection(vector_size: int):
    client = get_qdrant_client()
    collections = client.get_collections().collections
    existing = {collection.name for collection in collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def reset_collection(vector_size: int):
    client = get_qdrant_client()
    collections = client.get_collections().collections
    existing = {collection.name for collection in collections}
    if COLLECTION in existing:
        client.delete_collection(collection_name=COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def store_chunks(chunks, embeddings, metadata_list, replace=False):
    if not chunks:
        return
    client = get_qdrant_client()
    vector_size = len(embeddings[0])
    if replace:
        reset_collection(vector_size)
    else:
        ensure_collection(vector_size)

    points = []

    for chunk, emb, meta in zip(
        chunks,
        embeddings,
        metadata_list
    ):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                payload={
                    "text": chunk,
                    **meta
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )