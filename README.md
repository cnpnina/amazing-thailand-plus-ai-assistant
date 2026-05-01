# Thailand Tourism Mini Hackathon

RAG-based travel planner chatbot for following filming locations from TV series/movies.

The system combines:
- discovery (`Exa` first, `Gemini Search` fallback)
- crawling (`Crawl4AI`)
- chunking + embedding + vector retrieval (`Qdrant`)
- itinerary generation (`Gemini`)

Output is evidence-backed and includes source URLs, day-by-day routing, and guide-style storytelling.

## Core Features

- Exa-first web discovery with Gemini fallback when Exa results are missing/insufficient
- URL sanitization and blocked-domain filtering before crawl
- Crawl content extraction to markdown/text
- Per-run vector indexing (replace mode) to avoid cross-query contamination
- Retrieval with MMR reranking + keyword-aware filtering/backfill
- Planner guardrails (source URL validation, day-plan normalization, fallback itinerary)
- Multi-language support:
  - Detect input language (`th` / `en` / `zh`)
  - Render output in the same language as the user query
  - Guide-story fields support TH/EN/ZH internally

## Project Structure

- `chatbot_cli.py` - interactive CLI chatbot
- `orchestrator/prefect_flow.py` - Prefect tasks + flows
- `crawler/exa_discovery.py` - Exa discovery + Gemini search fallback + URL sanitation
- `crawler/crawl4ai_crawler.py` - crawler runtime
- `processing/cleaner.py` - text cleaning
- `processing/thai_chunk.py` - Thai-friendly chunking
- `processing/embedder.py` - embedding with fallback strategy
- `processing/retrieval.py` - vector retrieval + rerank + relevance backfill
- `processing/rerank.py` - MMR reranking
- `processing/route_planner.py` - Gemini planning + guardrails + rendering
- `storage/qdrant_store.py` - local Qdrant management

## Requirements

- Python 3.11+ (tested with 3.14 in this repo)
- Internet access for Exa, Gemini, and crawled websites

## Environment Variables

Create `.env`:

```bash
export EXA_API_KEY="your_exa_key"
export GOOGLE_API_KEY="your_google_key"        # or GEMINI_API_KEY
export GEMINI_MODEL="gemini-2.5-flash"
export GEMINI_SEARCH_MODEL="gemini-2.5-flash"  # optional; defaults to GEMINI_MODEL
export HF_TOKEN="your_hf_token_optional"
export EMBEDDING_MODEL="jinaai/jina-embeddings-v5-text-small"  # optional override
```

Notes:
- `GEMINI_MODEL` is required.
- `HF_TOKEN` is optional for public embedding models but recommended.
- Do not commit `.env` to GitHub.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run

Interactive chatbot:

```bash
source .venv/bin/activate
source .env
python chatbot_cli.py
```

Module entrypoint:

```bash
source .venv/bin/activate
source .env
python -m orchestrator.prefect_flow
```

## Pipeline Summary

1. `discover`:
   - Exa search
   - if low results -> Gemini search fallback
2. `crawl`: fetch web pages via Crawl4AI
3. `prepare_chunks`: clean + split + metadata
4. `index_chunks`: embed + upsert to local Qdrant (replace collection)
5. `retrieve_evidence`: query Qdrant + MMR rerank + relevance backfill
6. `plan_answer`: Gemini planner + guardrails + fallback rendering

## Guardrails

- Place must include `source_url` from retrieved evidence
- Day plan names are normalized/matched against validated places
- If day entries cannot be matched, auto split places by requested day count
- If model output is unusable, fallback plan is generated from evidence
- If evidence is too weak, return explicit “insufficient evidence”


## Troubleshooting

- Qdrant lock errors (local mode)
  - Avoid concurrent processes reading/writing the same `./.qdrant_local`

- Vector dimension mismatch
  - Index is rebuilt per run (`replace=True`), and retrieval includes dimension guards
