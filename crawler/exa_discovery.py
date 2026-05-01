from exa_py import Exa
import os
import re
import json
from google import genai
from google.genai import types
from pythainlp.tokenize import word_tokenize

exa = Exa(api_key=os.getenv("EXA_API_KEY"))
BLOCKED_DOMAINS = {
    "vertexaisearch.cloud.google.com",
    "tumblr.com",
}
BLOCKED_URL_PATTERNS = [
    r"/grounding-api-redirect/",
]

STOPWORDS = {
    "อยาก", "ช่วย", "หน่อย", "ตามรอย", "วางแพลน", "series", "ซีรีย์", "ซีรีส์",
    "trip", "travel", "plan", "footsteps", "can", "you", "help", "me", "the", "of",
    "สถานที่ถ่ายทำ", "โลเคชัน", "location", "real", "จริง",
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


def _relevance_score(doc: dict, keywords):
    haystack = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
    return sum(1 for keyword in keywords if keyword in haystack)


def _apply_relevance_filter(documents, relevance_query: str | None, min_keep: int = 3):
    keywords = _keywords_from_query(relevance_query or "")
    if not keywords:
        return documents

    scored = [(doc, _relevance_score(doc, keywords)) for doc in documents]
    scored.sort(key=lambda item: item[1], reverse=True)
    matched = [doc for doc, score in scored if score > 0]
    if len(matched) >= min_keep:
        return matched

    # Backfill from top-scored docs to avoid empty-result regressions.
    backfilled = []
    seen = set()
    for doc, _ in scored:
        url = doc.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        backfilled.append(doc)
        if len(backfilled) >= min_keep:
            break
    return backfilled if backfilled else matched


def _extract_json(text: str):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    return json.loads(raw)


def _extract_urls_from_text(text: str):
    return re.findall(r"https?://[^\s)>\"]+", text or "")


def _is_valid_content_url(url: str):
    if not url:
        return False
    lower_url = url.lower().strip()
    if not lower_url.startswith(("http://", "https://")):
        return False
    for domain in BLOCKED_DOMAINS:
        if domain in lower_url:
            return False
    for pattern in BLOCKED_URL_PATTERNS:
        if re.search(pattern, lower_url):
            return False
    return True


def _sanitize_documents(documents):
    sanitized = []
    seen = set()
    for doc in documents:
        url = (doc.get("url") or "").strip()
        if not _is_valid_content_url(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        sanitized.append(
            {
                "title": doc.get("title", ""),
                "url": url,
                "text": (doc.get("text") or "")[:5000],
            }
        )
    return sanitized


def _gemini_config():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, None
    model_name = os.getenv("GEMINI_SEARCH_MODEL") or os.getenv("GEMINI_MODEL")
    if not model_name:
        return api_key, "gemini-2.5-flash"
    return api_key, model_name


def discover_web_results(query: str, num_results=10, relevance_query: str | None = None, apply_filter: bool = True):
    try:
        result = exa.search_and_contents(
            query,
            num_results=num_results,
            text=True
        )
    except Exception:
        return []

    documents = []

    for item in result.results:
        documents.append({
            "title": item.title,
            "url": item.url,
            "text": item.text[:5000] if item.text else "",
        })

    documents = _sanitize_documents(documents)
    if apply_filter:
        documents = _apply_relevance_filter(documents, relevance_query or query, min_keep=min(3, num_results))

    return documents


def discover_with_gemini_search(query: str, num_results=10, relevance_query: str | None = None, apply_filter: bool = True):
    api_key, model_name = _gemini_config()
    if not api_key:
        return []

    client = genai.Client(api_key=api_key)
    prompt = f"""
Search the web and return at most {num_results} relevant pages for this query:
{query}

Return JSON only:
[
  {{"title":"...", "url":"https://...", "text":"short snippet"}}
]
""".strip()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception:
        return []

    documents = []
    try:
        data = _extract_json(response.text or "[]")
        for item in data:
            url = item.get("url")
            if not url:
                continue
            documents.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    "text": (item.get("text") or "")[:5000],
                }
            )
    except Exception:
        urls = _extract_urls_from_text(response.text or "")
        seen = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            documents.append({"title": "", "url": url, "text": ""})
            if len(documents) >= num_results:
                break

    documents = _sanitize_documents(documents)
    if apply_filter:
        return _apply_relevance_filter(documents, relevance_query or query)
    return documents


def discover_web_results_with_fallback(query: str, num_results=10, relevance_query: str | None = None, min_docs=3):
    docs = discover_web_results(
        query,
        num_results=num_results,
        relevance_query=relevance_query,
        apply_filter=True,
    )
    exa_raw_docs = discover_web_results(
        query,
        num_results=num_results,
        relevance_query=relevance_query,
        apply_filter=False,
    )
    if len(docs) >= min_docs:
        return docs[:num_results]

    gemini_docs = discover_with_gemini_search(
        query,
        num_results=num_results,
        relevance_query=relevance_query,
        apply_filter=True,
    )
    merged = docs + exa_raw_docs + gemini_docs

    deduped = []
    seen_urls = set()
    for doc in merged:
        url = doc.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(doc)
        if len(deduped) >= num_results:
            break

    if len(deduped) >= min_docs:
        return deduped

    # Last-resort fallback: allow unfiltered Gemini search results
    gemini_unfiltered = discover_with_gemini_search(
        query,
        num_results=num_results,
        relevance_query=relevance_query,
        apply_filter=False,
    )
    for doc in gemini_unfiltered:
        url = doc.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(doc)
        if len(deduped) >= num_results:
            break

    return deduped


def search_thai_history(query: str, num_results=10):
    # Backward-compatible alias.
    return discover_web_results_with_fallback(query, num_results=num_results, relevance_query=query)


if __name__ == "__main__":
    query = input("กรอกคำค้น: ").strip()
    if not query:
        raise ValueError("กรุณาระบุคำค้นก่อนเริ่มค้นหา")

    docs = discover_web_results_with_fallback(
        query
    )

    for d in docs:
        print(d["title"])
        print(d["url"])
        print("="*50)