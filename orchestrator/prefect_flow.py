import os

os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("PREFECT_SERVER_LOGGING_LEVEL", "WARNING")

from prefect import flow, task
from crawler.exa_discovery import discover_web_results_with_fallback
from crawler.crawl4ai_crawler import crawl_pages_sync
from processing.cleaner import clean_text
from processing.thai_chunk import semantic_chunk
from processing.metadata_extractor import extract_metadata
from processing.embedder import embed_texts
from processing.retrieval import retrieve
from processing.route_planner import build_route_plan_with_gemini
from storage.qdrant_store import store_chunks

@task
def discover(user_query: str, num_results=8):
    search_query = f"{user_query} สถานที่ถ่ายทำ โลเคชัน จริง"
    return discover_web_results_with_fallback(
        search_query,
        num_results=num_results,
        relevance_query=user_query,
    )

@task
def crawl(docs, max_pages=5):
    urls = []
    for doc in docs:
        url = (doc.get("url") or "").strip()
        if not url:
            continue
        if "vertexaisearch.cloud.google.com" in url or "/grounding-api-redirect/" in url:
            continue
        if "tumblr.com" in url:
            continue
        urls.append(url)
    return crawl_pages_sync(urls=urls, max_pages=max_pages)

@task
def prepare_chunks(crawled_pages):
    chunks = []
    metadata_list = []

    for page in crawled_pages:
        raw_text = page.get("markdown", "") or ""
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        page_chunks = semantic_chunk(cleaned, max_chars=1000)

        for chunk in page_chunks:
            if len(chunk) < 80:
                continue
            metadata = extract_metadata(chunk)
            metadata["source_url"] = page.get("url")
            metadata["title"] = page.get("title")
            chunks.append(chunk)
            metadata_list.append(metadata)

    return chunks, metadata_list

@task
def index_chunks(chunks, metadata_list):
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    # Replace index each run to avoid cross-query contamination.
    store_chunks(chunks, embeddings, metadata_list, replace=True)
    return len(chunks)

@task
def retrieve_evidence(user_query: str, top_k=12):
    return retrieve(user_query, candidate_limit=24, top_k=top_k)

@task
def plan_answer(user_query: str, evidences):
    result = build_route_plan_with_gemini(user_query=user_query, evidences=evidences)
    return result["answer_text"]

@flow
def travel_chatbot_flow(user_query: str):
    docs = discover(user_query)
    crawled_pages = crawl(docs)
    chunks, metadata_list = prepare_chunks(crawled_pages)
    index_chunks(chunks, metadata_list)
    evidences = retrieve_evidence(user_query)
    return plan_answer(user_query, evidences)

@flow
def ingestion_pipeline(user_query: str):
    docs = discover(user_query)
    crawled_pages = crawl(docs)
    chunks, metadata_list = prepare_chunks(crawled_pages)
    indexed_count = index_chunks(chunks, metadata_list)
    return f"Indexed {indexed_count} chunks"


@flow
def query_pipeline(user_query: str):
    evidences = retrieve_evidence(user_query)
    return plan_answer(user_query, evidences)

if __name__ == "__main__":
    question = input("ถามได้เลย: ").strip()
    if not question:
        raise ValueError("กรุณาระบุคำถามก่อนเริ่มรัน flow")
    answer = travel_chatbot_flow(question)
    print(answer)