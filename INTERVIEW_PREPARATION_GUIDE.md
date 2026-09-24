# 🎓 Enterprise Hybrid RAG Engine — Simple English Technical Interview Guide

> **Project Overview**: An end-to-end Hybrid RAG (Retrieval-Augmented
> Generation) system that searches internal documents using both Keyword Search
> (BM25) and Vector Search (ChromaDB), merges them using Reciprocal Rank Fusion
> (RRF), reranks the best passages with a Cross-Encoder, generates grounded
> answers with Groq / Llama 3.1, and automatically verifies citations using
> Python code to stop hallucinations.

---

## Quick Navigation

1. [Architecture-Level Questions](#1-architecture-level-questions)
2. [File-by-File Code Deep-Dive](#2-file-by-file-code-deep-dive)
3. [Design Decision Questions ("Why did you choose this?")](#3-design-decision-questions-why-did-you-choose-this)
4. [Limitations and Production Trade-Offs](#4-limitations-and-production-trade-offs)
5. [Step-by-Step Code Walkthroughs](#5-step-by-step-code-walkthroughs)
6. [Core AI & NLP Concepts in Simple English](#6-core-ai--nlp-concepts-in-simple-english)

---

# 1. ARCHITECTURE-LEVEL QUESTIONS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         High-Level Architecture                             │
└─────────────────────────────────────────────────────────────────────────────┘

  [Documents] ──► [Clean Text] ──► [Chunking] ──► [Deduplication]
                                                        │
                      ┌─────────────────────────────────┴─────────────────┐
                      ▼                                                   ▼
             [BM25 Keyword Index]                                [ChromaDB Vectors]
                      │                                                   │
                      └─────────────────┬─────────────────────────────────┘
                                        ▼
                           [Reciprocal Rank Fusion (RRF)]
                                        │
                                        ▼
                           [Cross-Encoder Reranker]
                                        │
                                        ▼
                           [Groq / Llama 3.1 Generator]
                                        │
                                        ▼
                           [Citation Verifier Guardrail]
                                        │
                                        ▼
                                [Verified Answer]
```

### Q1. What is the overall architecture of your project, and why did you divide it into separate modules?

**Simple Spoken Answer (30–45s):**

> _"Our project has a 5-step pipeline:_\
> _1. **Ingestion & Cleaning**: We read PDFs, Markdown, HTML, and text files,
> clean messy characters, and remove duplicate chunks._\
> _2. **Dual Indexing**: We store chunks in two places: BM25 for keyword
> matching and ChromaDB for vector/semantic matching._\
> _3. **Two-Stage Retrieval**: When a question comes in, both search engines
> find candidate chunks, merge them using Reciprocal Rank Fusion (RRF), and a
> Cross-Encoder picks the top 5 most relevant chunks._\
> _4. **Answer Generation**: Groq (Llama-3.1) writes a grounded answer in strict
> JSON format with citations._\
> _5. **Citation Verification**: Python code verifies that every citation in the
> answer actually exists in the source text._\
> _We separated these into independent modules so we can easily swap any
> part—like changing the vector database or upgrading the embedding
> model—without breaking the rest of the application."_

---

### Q2. How does the system work in both "Standalone Mode" and "Microservice Mode"?

**Simple Spoken Answer (30–45s):**

> _"In `src/ui/app.py`, the code checks the `BACKEND_API_URL` setting:_\
> _- **Microservice Mode**: If a backend URL is provided (like in Docker Compose
> where `BACKEND_API_URL=http://rag-api:8000`), Streamlit only acts as a
> frontend UI and sends HTTP requests (`/v1/ingest`, `/v1/ask`) to the FastAPI
> backend._\
> _- **Standalone Mode**: If no backend URL is set, Streamlit loads the models
> directly into its own memory (`st.session_state`). This lets us run the whole
> app with just one command (`python run.py`) for quick local testing without
> starting a separate server."_

---

### Q3. Why do you use a two-stage retrieval pipeline instead of just asking the LLM directly?

**Simple Spoken Answer (45s):**

> _"If we only use single-stage vector search, we face two big problems: first,
> vector search often misses exact numbers, policy codes, or short
> abbreviations. Second, standard vector search compares queries and documents
> separately, without looking at how individual words relate to each other._\
> _Our two-stage approach solves this:_\
> _- **Stage 1 (High Recall)**: BM25 and ChromaDB fetch 20 candidate chunks in
> parallel to make sure we don't miss anything._\
> _- **Stage 2 (High Precision)**: A Cross-Encoder model reads the query and the
> 20 chunks together, compares them word-by-word, and selects the top 5 best
> chunks._\
> _This gives us the best of both worlds: we don't miss relevant documents, and
> the LLM receives only the most accurate context."_

---

### Q4. How do you manage model loading and application state in FastAPI?

**Simple Spoken Answer (30s):**

> _"In `src/main.py`, we use FastAPI's `lifespan` manager. When the server
> starts up, it loads the embedding model, the BM25 index file, the
> Cross-Encoder, and the Groq client once and attaches them to `app.state`._\
> _This means we don't reload heavy ML models on every API request. Requests are
> served instantly in just a few milliseconds, and multiple requests can share
> the same models safely."_

---

### Q5. How would you scale this system to handle 1 million+ documents?

**Simple Spoken Answer (45s):**

> _"To scale to 1 million documents, I would make three key changes:_\
> _1. **Cloud Vector Database**: Replace local ChromaDB with a distributed
> vector database like Qdrant or Milvus that can shard data across multiple
> servers._\
> _2. **Distributed Keyword Search**: Replace the in-memory BM25 pickle file
> with Elasticsearch or OpenSearch._\
> _3. **Background Job Queue**: Instead of processing uploaded documents inside
> the HTTP request, use Celery or Redis queues so background workers handle
> parsing and embedding asynchronously."_

---

### Q6. What is the Citation Verification Guardrail, and why is it built in Python rather than just asking the LLM?

**Simple Spoken Answer (30–45s):**

> _"Even if you instruct an LLM to add citations like `[1]` or `[2]`, it can
> still hallucinate non-existent numbers (for example, citing `[4]` when we only
> gave it 2 chunks)._\
> _Our `CitationVerifier` is a deterministic Python script. It uses regular
> expressions to find all bracketed numbers, checks if they match real chunks in
> the retrieved list, and ensures those chunks are not empty. If an invalid
> citation is found, it immediately flags it as an issue in the API response."_

---

### Q7. How is Docker set up for this project?

**Simple Spoken Answer (30s):**

> _"We use a clean `Dockerfile` based on `python:3.12-slim`. For security, we
> create a non-root user (`useradd -m -u 1000`) and install CPU-only PyTorch to
> keep the image lightweight._\
> _Using `docker-compose.yaml`, we run two services: `rag-api` (FastAPI backend
> on port 8000) and `rag-ui` (Streamlit frontend on port 8501). Both share a
> mounted `./data` folder so they access the exact same database and index
> files."_

---

### Q8. How does the automated evaluation (LLM-as-a-Judge) work?

**Simple Spoken Answer (30–45s):**

> _"In `src/evaluation/metrics_runner.py`, we test the pipeline against a test
> dataset of 50+ real questions (`golden_dataset.json`)._\
> _We use a large judge model (`llama-3.3-70b-versatile` on Groq) to grade two
> main scores from 0.0 to 1.0:_\
> _- **Faithfulness**: Is every fact in the answer backed by the retrieved
> context (no hallucinations)?_\
> _- **Answer Relevancy**: Does the answer directly answer the user's question
> without useless fluff?_\
> _This allows us to run automated quality checks before releasing changes."_

---

### Q9. What happens if the database is empty or the LLM fails?

**Simple Spoken Answer (30s):**

> _"We have defensive error handling at every step:_\
> _- If a user asks a question before any documents are uploaded, the API
> immediately returns a friendly message saying the index is empty, without
> wasting money on LLM calls._\
> _- If the LLM returns invalid JSON or hits an API rate limit, the generator
> catches the error and returns a clean fallback error object._\
> _- If a document has strange characters, our unicode cleaner removes them so
> the system never crashes."_

---

### Q10. How do you keep BM25 and ChromaDB in sync when new files are uploaded?

**Simple Spoken Answer (30s):**

> _"When a document is uploaded, it is split into chunks and deduplicated once.
> The exact same list of unique chunks (with IDs like `doc1-chunk-0`,
> `doc1-chunk-1`) is sent to both `sparse_index.index_chunks()` and
> `dense_index.index_chunks()`._\
> _BM25 adds the new tokens and saves the pickle file, while ChromaDB runs
> `collection.upsert()`. Because both use the exact same chunk IDs, both search
> indexes stay 100% synchronized."_

---

# 2. FILE-BY-FILE CODE DEEP-DIVE

```
Project Files:
├── src/
│   ├── config.py             ── Hyperparameters & directory paths
│   ├── main.py               ── FastAPI REST API routes (/v1/ingest, /v1/ask)
│   ├── ingestion/
│   │   ├── schemas.py        ── Pydantic models for Document and Chunk
│   │   ├── parsers.py        ── PDF, HTML, MD, TXT parsers + unicode cleaner
│   │   ├── chunkers.py       ── Fixed-size sliding window & Markdown chunkers
│   │   └── deduplicator.py   ── Cosine similarity chunk deduplicator
│   ├── indexing/
│   │   ├── sparse.py         ── BM25Okapi keyword search & persistence
│   │   ├── dense.py          ── ChromaDB vector storage & search
│   │   └── hybrid_retriever.py ── Combines BM25 & ChromaDB using RRF
│   ├── reranking/
│   │   └── cross_encoder.py  ── Neural reranking using ms-marco-MiniLM
│   ├── generation/
│   │   ├── generator.py      ── Groq Llama 3.1 prompt & JSON mode output
│   │   └── verifier.py       ── Regex-based citation validator
│   ├── evaluation/
│   │   └── metrics_runner.py ── Automated LLM-as-a-Judge test runner
│   └── ui/
│       └── app.py            ── Streamlit web dashboard
```

---

### 1. `src/config.py`

#### Q: What is the job of `src/config.py` and why do we use `Path(__file__).resolve().parent.parent`?

- **Code Reference**: Lines 11–16
- **Simple Answer**:
  > _"`AppConfig` is the single place where all settings live (chunk sizes,
  > model names, folder paths). Using `Path(__file__).resolve().parent.parent`
  > gives us the absolute path to the project root. This ensures that paths like
  > `data/chroma_db` and `data/sparse_index.pkl` work reliably whether you run
  > the app from the root folder, from inside `src/`, or inside a Docker
  > container."_

#### Q: What are the main hyperparameters defined here?

- **Code Reference**: Lines 18–28
  - `CHUNK_SIZE = 1500` characters (~300 words).
  - `CHUNK_OVERLAP = 300` characters (20% overlap).
  - `RETRIEVAL_TOP_K = 10` (candidates from each searcher).
  - `RERANK_TOP_N = 5` (top passages sent to LLM).
- **Simple Answer**:
  > _"1500 characters keeps full policy rules together in one piece. 300
  > characters of overlap prevents sentences from being split in half at the
  > boundaries. In stage 1, we pull 10 results from BM25 and 10 from ChromaDB
  > (20 total), and the reranker narrows them down to the top 5 to keep the LLM
  > fast and focused."_

---

### 2. `src/ingestion/schemas.py`

#### Q: How do `DocumentMetadata` and `ChunkMetadata` track where data came from?

- **Code Reference**: Lines 8–43
- **Simple Answer**:
  > _"`ChunkMetadata` inherits from `DocumentMetadata`. It adds `chunk_index`
  > and `parent_document_id`. This means every single text chunk remembers its
  > original file path, file format, creation date, and exact position. We use
  > this metadata for citation verification and debugging."_

#### Q: Why do chunk IDs look like `f"{document.id}-chunk-{chunk_index}"`?

- **Simple Answer**:
  > _"Because it creates a predictable, unique ID. If you upload the same
  > document again, ChromaDB's `upsert` simply updates the existing chunk ID
  > instead of creating duplicate vectors."_

---

### 3. `src/ingestion/parsers.py`

#### Q: What problems does `sanitize_unicode_string()` fix?

- **Code Reference**: Lines 15–42
- **Simple Answer**:
  > _"When extracting text from PDFs, 4 common bugs happen that can crash your
  > app:_\
  > _1. **Unpaired Surrogates (`\ud800` to `\udfff`)**: These break Pydantic's
  > internal Rust string validator._\
  > _2. **Null Bytes (`\x00`)**: These crash database queries in SQLite and
  > ChromaDB._\
  > _3. **Broken Bullets (``)**: Weird symbols from PDF fonts get converted
  > into clean standard dashes (`\n -`)._\
  > _4. **Glued Words**: PDF column text often glues words together (like
  > `'laiddownby'`). Our regex splits them into `'laid down by'`._\
  > _This cleaner guarantees that only safe, clean text reaches the embedding
  > models."_

#### Q: How does `process_file()` handle different file formats?

- **Code Reference**: Lines 70–94
- **Simple Answer**:
  > _"It checks the file extension: `.pdf` uses `pypdf.PdfReader`, `.html` uses
  > `BeautifulSoup` to strip HTML tags, and `.md` or `.txt` are read as UTF-8
  > text. If an unknown file format is passed, it logs a warning and safely
  > falls back to plain text reading."_

---

### 4. `src/ingestion/chunkers.py`

#### Q: How does `fixed_size_chunk()` work, and how does it prevent infinite loops?

- **Code Reference**: Lines 11–46
- **Simple Answer**:
  > _"It moves a window across the text. The jump size is
  > `step = chunk_size - chunk_overlap` ($1500 - 300 = 1200$). It extracts
  > chunks of 1500 characters, then moves forward by 1200 characters. If someone
  > accidentally sets overlap greater than chunk size, `step` would become 0 or
  > negative, causing an infinite loop. We added
  > `if step <= 0: step = chunk_size` as a safety guard to prevent that."_

#### Q: What makes `structure_aware_markdown_chunk()` smart?

- **Code Reference**: Lines 49–82
- **Simple Answer**:
  > _"Instead of cutting Markdown text at fixed character counts, it splits at
  > Markdown headers (`# Header 1`, `## Header 2`). It attaches the header title
  > to the beginning of each chunk (`f"{current_header}\n\n{section}"`). When
  > the vector database indexes the chunk, it knows both the section title and
  > the content underneath it."_

---

### 5. `src/ingestion/deduplicator.py`

#### Q: How does `ChunkDeduplicator` work, and why does it look at only the last 50 vectors?

- **Code Reference**: Lines 24–62
- **Simple Answer**:
  > _"It generates dense embeddings for all chunks, then compares each chunk
  > against previously accepted chunks. If the cosine similarity is higher than
  > 0.95, it drops the chunk as a duplicate._\
  > _Instead of comparing every chunk to all previous chunks ($O(N^2)$), it only
  > checks the last 50 vectors (`seen_vectors[-50:]`). In real documents,
  > duplicate content (like repeated headers, tables, or disclaimers) is grouped
  > in nearby sections. The 50-chunk window catches these duplicates while
  > keeping the code fast."_

---

### 6. `src/indexing/sparse.py`

#### Q: How does BM25 tokenize text and save to disk?

- **Code Reference**: Lines 42–49 & 94–123
- **Simple Answer**:
  > _"`_tokenize()` turns text to lowercase, uses a regex to grab words and
  > hyphenated terms (like `wi-fi`), and removes common English stopwords
  > (`the`, `is`, `at`, `for`)._\
  > _For storage, it uses Python's `pickle` library to save both the tokenized
  > chunk list and the `BM25Okapi` model into `data/sparse_index.pkl`."_

#### Q: How does `index_chunks()` avoid duplicate indexing?

- **Code Reference**: Lines 50–70
- **Simple Answer**:
  > _"It creates a set of existing chunk IDs (`existing_ids`), filters out any
  > chunk that is already in the index, appends only the new ones, re-trains the
  > BM25 model, and saves the new pickle file to disk."_

---

### 7. `src/indexing/dense.py`

#### Q: How does `LocalSentenceTransformerEmbeddingFunction` connect to ChromaDB?

- **Code Reference**: Lines 13–27
- **Simple Answer**:
  > _"It implements ChromaDB's `EmbeddingFunction` interface. When created, it
  > checks if a GPU is available (`cuda` or `cpu`) and loads `all-MiniLM-L6-v2`.
  > ChromaDB calls this function automatically to convert text lists into
  > 384-dimensional float vectors."_

#### Q: Why does `search()` do `similarity_score = float(1.0 - distance)`?

- **Code Reference**: Lines 73–76
- **Simple Answer**:
  > _"ChromaDB uses Cosine Distance ($1 - \text{similarity}$), where 0.0 means
  > identical and 2.0 means opposite. To convert this into an intuitive
  > similarity score where 1.0 is the best match, we calculate
  > `1.0 - distance`."_

---

### 8. `src/indexing/hybrid_retriever.py`

#### Q: Explain how Reciprocal Rank Fusion (RRF) works in `retrieve()`.

- **Code Reference**: Lines 14–59
- **Simple Answer**:
  > _"We ask BM25 for top 20 chunks and ChromaDB for top 20 chunks. For every
  > chunk $d$, RRF calculates:_\
  > $$\text{Score} = \sum \frac{1}{60 + \text{rank}}$$\
  > _If a chunk appears near the top in both BM25 and ChromaDB, its RRF score
  > becomes very high. We sort by this score and return the top 10 merged
  > results. We also record whether each chunk was found by `sparse`, `dense`,
  > or `both`."_

---

### 9. `src/reranking/cross_encoder.py`

#### Q: How does `DocumentReranker.rerank()` pick the best chunks?

- **Code Reference**: Lines 20–46
- **Simple Answer**:
  > _"It takes the user's question and pairs it with each candidate chunk:
  > `[[query, chunk_1], [query, chunk_2], ...]`. It runs these pairs through
  > `cross-encoder/ms-marco-MiniLM-L-6-v2`, which compares every word in the
  > query against every word in the chunk using full cross-attention. It sorts
  > them by relevance score and returns the top 5 elite chunks."_

---

### 10. `src/generation/generator.py`

#### Q: How does `GroundedGenerator` prevent hallucinations?

- **Code Reference**: Lines 17–57
- **Simple Answer**:
  > _"It formats the top chunks into numbered blocks
  > (`--- CONTEXT BLOCK [1] ---`, etc.). Then it calls Groq's Llama-3.1 model
  > with two strict settings:_\
  > _1. `temperature=0.0`: Forces the model to be completely deterministic
  > without creative guessing._\
  > _2. `response_format={"type": "json_object"}`: Forces the model to return
  > valid JSON._\
  > _The prompt mandates that every factual claim must end with a citation like
  > `[1]`, and if the context does not have the answer, it must set
  > `"is_context_sufficient": false`."_

---

### 11. `src/generation/verifier.py`

#### Q: How does `CitationVerifier.verify_citations()` check citation accuracy?

- **Code Reference**: Lines 9–47
- **Simple Answer**:
  > _"It uses regex `\[(\d+)\]` to find all citation numbers in the generated
  > text. For each number, it checks:_\
  > _- Is the number within range? (If the LLM cited `[4]` but only 2 chunks
  > were provided, it flags `MALFORMED_INDEX`)._\
  > _- Is the cited chunk empty? (If yes, it flags `EMPTY_CONTEXT`)._\
  > _If all citations point to valid chunks, it returns `is_valid: true`."_

---

### 12. `src/evaluation/metrics_runner.py`

#### Q: How does `RAGMetricsRunner` grade the pipeline?

- **Code Reference**: Lines 41–63
- **Simple Answer**:
  > _"It takes test questions from `data/golden_dataset.json` and runs them
  > through the full pipeline. Then it uses `llama-3.3-70b-versatile` as an
  > impartial judge to score two metrics:_\
  > _- **Faithfulness**: Are all statements in the answer backed by the
  > context?_\
  > _- **Answer Relevancy**: Does the answer directly address the user's
  > question?_\
  > _It pauses for 3 seconds (`time.sleep(3)`) between questions so we don't
  > exceed Groq's free rate limits."_

---

### 13. `src/ui/app.py`

#### Q: Why does `src/ui/app.py` run a "warm-up" on startup?

- **Code Reference**: Lines 146–163
- **Simple Answer**:
  > _"When PyTorch models run for the first time, loading weights into memory
  > causes a 1–2 second delay (cold start). We pass a dummy text through the
  > embedding and reranker models during app startup so the first real user
  > query gets an instant response."_

---

### 14. `src/main.py`

#### Q: How do `/v1/ingest` and `/v1/ask` handle input validation?

- **Code Reference**: Lines 78–84 & 262–370
- **Simple Answer**:
  > _"Both endpoints use Pydantic models with `Field(..., min_length=1)` to
  > reject empty strings with HTTP 422, and explicit `.strip()` checks to reject
  > pure whitespace with HTTP 400. If an uploaded file doesn't exist, it returns
  > HTTP 404. If `/v1/ask` is called when no documents have been indexed, it
  > gracefully returns `is_context_sufficient: false` instead of crashing."_

---

# 3. DESIGN DECISION QUESTIONS ("WHY DID YOU CHOOSE THIS?")

| Question                                                                | Simple, Defensible Answer                                                                                                                                                                                                                                                                                                               |
| :---------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Why use both BM25 and ChromaDB?**                                     | Vector embeddings understand concepts (e.g., matching "leave rules" with "vacation policy"), but they struggle with exact keywords, policy codes ("Section 4.1"), or acronyms. BM25 is great at exact keywords. Combining both ensures we never miss a relevant document.                                                               |
| **Why Reciprocal Rank Fusion (RRF) instead of adding scores together?** | BM25 scores can be any positive number ($0$ to $50+$), while cosine similarity is always between $0$ and $1$. Normalizing them is unreliable and breaks when query lengths change. RRF only cares about the **rank position** ($\frac{1}{60 + \text{rank}}$), so it works consistently without any manual tuning.                       |
| **Why add a Cross-Encoder Reranker?**                                   | Vector search (Bi-Encoder) encodes the query and document separately, so it cannot compare individual words against each other. The Cross-Encoder reads both together with full cross-attention. It is too slow to run on 1,000 documents, but running it on our top 20 candidates takes only 20ms and significantly improves accuracy. |
| **Why 1500 character chunk size with 300 overlap?**                     | In internal documents, rules and policy clauses usually span 2–3 paragraphs (~300 words, or ~1500 characters). A smaller chunk size (like 400 chars) cuts rules off from their conditions. The 300-character overlap (20%) ensures sentences at chunk borders aren't cut in half.                                                       |
| **Why Deduplicate Chunks (0.95 similarity)?**                           | Real documents often contain repeated footers, boilerplate disclaimers, or duplicate tables. Dropping chunks that are $>95\%$ identical saves database storage and prevents the LLM from reading the exact same paragraph twice.                                                                                                        |
| **Why Groq with Llama-3.1, JSON Mode, and Temp 0.0?**                   | Groq generates 500–800 tokens per second, giving us sub-second response times. Temperature 0.0 ensures the output is predictable and not made up. JSON mode ensures the frontend UI can reliably parse the answer every single time.                                                                                                    |
| **Why verify citations with Python code instead of asking the LLM?**    | Asking an LLM "Did you cite correctly?" is slow, costs extra tokens, and the LLM can hallucinate its self-check. A Python regex check runs in less than 1 millisecond, costs zero tokens, and is 100% mathematically reliable.                                                                                                          |

---

# 4. LIMITATIONS AND PRODUCTION TRADE-OFFS

### Q1: What is the downside of saving BM25 as a local pickle file (`sparse_index.pkl`)?

> **Simple Answer (45s):**\
> _"Our BM25 index stores all words in memory and saves them as a Python pickle
> file:_\
> _1. **Memory Limits**: The entire index lives in RAM. If we scale to millions
> of documents, RAM will run out._\
> _2. **Re-indexing Overhead**: Every time a new document is added, `rank-bm25`
> re-tokenizes the entire corpus from scratch._\
> _3. **Production Fix**: In a large production system, we would replace this
> with Elasticsearch or OpenSearch, which store inverted indexes on disk and
> support instant incremental updates."_

---

### Q2: What is the limitation of local ChromaDB in a multi-server setup?

> **Simple Answer (30s):**\
> _"Local ChromaDB uses SQLite on the local hard drive. If we run 4 backend
> server containers behind a load balancer, they cannot safely write to the same
> local SQLite file simultaneously without database locking errors.\
> **Production Fix**: Run ChromaDB in Client-Server mode or use a cloud vector
> database like Qdrant or Pinecone."_

---

### Q3: What is the trade-off of the 50-chunk deduplication window?

> **Simple Answer (30s):**\
> _"Our deduplicator only checks the previous 50 chunks. If the exact same
> disclaimer appears on page 1 and again on page 100 (more than 50 chunks
> apart), the system won't catch it.\
> **Production Fix**: Use Locality-Sensitive Hashing (LSH) or MinHash for
> global, corpus-wide duplicate detection in $O(1)$ time."_

---

### Q4: Does the Citation Verifier check if the text actually supports the claim?

> **Simple Answer (30s):**\
> _"Our current verifier checks **structural validity**—it makes sure the cited
> index number exists in our retrieved list and is not empty. However, it does
> not perform deep logic analysis to prove the sentence is true.\
> **Production Fix**: Add a lightweight Natural Language Inference (NLI) model
> (like `roberta-large-mnli`) to verify factual entailment between the generated
> sentence and the cited passage."_

---

### Q5: What happens if a user uploads a scanned image PDF?

> **Simple Answer (30s):**\
> _"`pypdf` only reads digital text. If someone uploads a scanned image PDF,
> `pypdf` returns an empty string.\
> **Production Fix**: Add an OCR tool (like Tesseract or AWS Textract) that
> triggers automatically when extracted text length is near zero."_

---

### Q6: How does the system handle multi-turn conversations (chat memory)?

> **Simple Answer (30s):**\
> _"Right now, `/v1/ask` is single-turn (it treats every question
> independently). If a user asks a follow-up like 'What is its fee?', the search
> engine searches for the word 'its' and returns poor results.\
> **Production Fix**: Add a Query Rewriter step using an LLM that reads the chat
> history and transforms 'What is its fee?' into 'What is the fee for the
> student assistance scheme?' before searching."_

---

# 5. STEP-BY-STEP CODE WALKTHROUGHS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          End-to-End System Flows                            │
└─────────────────────────────────────────────────────────────────────────────┘

  Flow 1: Ingestion
  Upload ──► Clean Unicode ──► Split 1500 Chars ──► Dedup (>0.95) ──► BM25 + ChromaDB

  Flow 2: Retrieval & Rerank
  Query ──► BM25 (Top 20) + ChromaDB (Top 20) ──► RRF (Top 10) ──► Cross-Encoder (Top 5)

  Flow 3: Generation & Verification
  Top 5 Chunks ──► Groq Llama 3.1 ──► JSON Answer with [1] ──► Regex Citation Check ──► UI
```

---

### Walkthrough 1: Document Ingestion Flow (From Upload to Indexing)

1. **User Action**: The user selects a file on the Streamlit dashboard or sends
   a POST request to `/v1/ingest` with `{"file_path": "..."}`
   ([src/main.py:262](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/main.py#L262)).
2. **Parsing & Cleaning**: `DocumentParserRouter.process_file()`
   ([src/ingestion/parsers.py:70](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/ingestion/parsers.py#L70))
   extracts raw text from PDF, HTML, or TXT, and runs
   `sanitize_unicode_string()` to strip null bytes and fix glued words.
3. **Chunking**:
   - If it's a Markdown file (`.md`), `structure_aware_markdown_chunk()` splits
     by `#` headers and prepends section names.
   - For all other files, `fixed_size_chunk()` splits text into 1500-character
     windows with a 300-character sliding overlap
     ([src/ingestion/chunkers.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/ingestion/chunkers.py)).
4. **Deduplication**: `ChunkDeduplicator.deduplicate()`
   ([src/ingestion/deduplicator.py:24](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/ingestion/deduplicator.py#L24))
   embeds the chunks and removes any chunk that has a cosine similarity $>0.95$
   with recent chunks.
5. **Dual Index Storage**:
   - `SparseBM25Index.index_chunks()`
     ([src/indexing/sparse.py:50](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/sparse.py#L50))
     tokenizes the text and saves the BM25 model to `data/sparse_index.pkl`.
   - `DenseVectorIndex.index_chunks()`
     ([src/indexing/dense.py:43](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/dense.py#L43))
     embeds the chunks and upserts them into ChromaDB's HNSW vector index.
6. **Result**: Returns `{"status": "success", "chunks_indexed": N}`.

---

### Walkthrough 2: Hybrid Search & Neural Reranking Flow

1. **User Action**: The user types a question into the search box or sends a
   POST request to `/v1/ask`
   ([src/main.py:315](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/main.py#L315)).
2. **Parallel Retrieval**: `HybridRetriever.retrieve()`
   ([src/indexing/hybrid_retriever.py:14](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/hybrid_retriever.py#L14))
   runs two searches in parallel:
   - BM25 finds the top 20 keyword-matching chunks.
   - ChromaDB finds the top 20 semantic vector matches.
3. **RRF Merging**: The system applies the formula
   $\text{Score} = \sum \frac{1}{60 + \text{rank}}$ to combine the results,
   sorts them, and keeps the top 10 candidates.
4. **Cross-Encoder Reranking**: `DocumentReranker.rerank()`
   ([src/reranking/cross_encoder.py:20](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/reranking/cross_encoder.py#L20))
   passes the query and all 10 candidates through `ms-marco-MiniLM-L-6-v2`,
   scores them using cross-attention, and returns the top 5 elite passages.

---

### Walkthrough 3: Grounded Generation & Citation Verification Flow

1. **Context Assembly**: `GroundedGenerator.generate_answer()`
   ([src/generation/generator.py:34](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/generation/generator.py#L34))
   formats the top 5 chunks into numbered blocks:
   ```
   --- CONTEXT BLOCK [1] ---
   <chunk 1 text>
   --- CONTEXT BLOCK [2] ---
   <chunk 2 text>
   ```
2. **LLM Generation**: It calls Groq (`llama-3.1-8b-instant`) with
   `temperature=0.0` and JSON mode. The LLM returns a structured JSON answer
   where every statement is cited with bracketed numbers like `[1]`.
3. **Citation Verification**: `CitationVerifier.verify_citations()`
   ([src/generation/verifier.py:9](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/generation/verifier.py#L9))
   extracts the citation numbers using regex `\[(\d+)\]` and verifies that each
   cited index exists in the retrieved chunk list.
4. **Display**: The final answer and citation verification badge are returned to
   the user.

---

# 6. CORE AI & NLP CONCEPTS IN SIMPLE ENGLISH

```
Bi-Encoder (Fast Vector Search)             Cross-Encoder (Accurate Reranker)
┌──────────┐        ┌──────────┐            ┌────────────────────────────────┐
│  Query   │        │ Document │            │       [Query + Document]       │
└────┬─────┘        └────┬─────┘            └───────────────┬────────────────┘
     ▼                   ▼                                  ▼
[ Vector A ]        [ Vector B ]                 [ Full Cross-Attention ]
     └───────► Dot ◄─────┘                                  │
             Product                                        ▼
                │                                     [ Match Score ]
                ▼
          [ Quick Score ]
```

---

### Concept 1: Dense Embeddings & Vector Space

- **Simple Theory**: An embedding model converts a piece of text into a list of
  numbers (a vector). Sentences with similar meanings end up close together in
  this mathematical space, even if they use completely different words (e.g.,
  "automobile" and "car").
- **In Our Project**: We use `all-MiniLM-L6-v2`
  ([src/indexing/dense.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/dense.py)),
  which converts each text chunk into a vector of 384 numbers. ChromaDB uses an
  HNSW graph to search these vectors in milliseconds.

---

### Concept 2: Cosine Similarity vs Cosine Distance

- **Simple Theory**:
  - **Cosine Similarity**: Measures the angle between two vectors. $1.0$ means
    identical direction (same meaning), and $0.0$ means completely unrelated.
  - **Cosine Distance**: How far apart two vectors are
    ($1.0 - \text{Similarity}$). $0.0$ means identical, and $1.0+$ means far
    apart.
- **In Our Project**: In
  [src/ingestion/deduplicator.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/ingestion/deduplicator.py),
  we calculate Cosine Similarity with NumPy. ChromaDB returns Cosine Distance,
  so in
  [src/indexing/dense.py:75](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/dense.py#L75)
  we calculate `1.0 - distance` to get back the similarity score.

---

### Concept 3: Bi-Encoder vs Cross-Encoder

- **Simple Theory**:
  - **Bi-Encoder**: Encodes the query and the document into separate vectors
    independently. It is extremely fast because document vectors can be computed
    ahead of time, but it misses fine-grained word interactions.
  - **Cross-Encoder**: Feeds the query and document together into the
    Transformer. Every word in the query attends to every word in the document
    simultaneously. It is much more accurate, but too slow to run on large
    databases.
- **In Our Project**: We use the **Bi-Encoder** (`all-MiniLM-L6-v2`) in Stage 1
  to quickly scan thousands of chunks, and the **Cross-Encoder**
  (`ms-marco-MiniLM-L-6-v2`) in Stage 2 to re-score the top 10 candidates
  ([src/reranking/cross_encoder.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/reranking/cross_encoder.py)).

---

### Concept 4: BM25Okapi Algorithm (Keyword Search)

- **Simple Theory**: BM25 is an improved version of TF-IDF. It scores documents
  based on how often query words appear in the document, but adds two smart
  limits:
  1. It prevents a word repeated 50 times from dominating the score (Term
     Frequency saturation).
  2. It adjusts for document length so long documents don't get an unfair
     advantage.
- **In Our Project**: We use `BM25Okapi` in
  [src/indexing/sparse.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/sparse.py)
  to reliably find exact product names, section numbers, and technical
  abbreviations.

---

### Concept 5: Reciprocal Rank Fusion (RRF)

- **Simple Theory**: When you have search results from two different search
  engines (like BM25 and ChromaDB), their raw scores cannot be directly added
  together. RRF solves this by scoring items based only on their **ranking
  position**: $$\text{Score} = \sum \frac{1}{60 + \text{rank}}$$
- **In Our Project**: Implemented in
  [src/indexing/hybrid_retriever.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/indexing/hybrid_retriever.py),
  RRF smoothly merges candidate lists without requiring any manual score
  normalization.

---

### Concept 6: The RAG Triad & LLM-as-a-Judge

- **Simple Theory**: The RAG Triad measures the 3 main quality checks for any
  RAG system:
  1. **Context Relevance**: Did we find the right documents for the question?
  2. **Faithfulness**: Is the generated answer 100% supported by the documents
     (no hallucinations)?
  3. **Answer Relevancy**: Does the answer directly answer the user's question?
- **In Our Project**: In
  [src/evaluation/metrics_runner.py](file:///c:/Users/yadav/OneDrive/Desktop/extra/GEN%20AI%20projects/Internship/RAG%20Pipeline%20with%20Hybrid%20Search%20Over%20Internal%20Docs/RAG/Advanced-NLP-Search-Retrieval-Engine/src/evaluation/metrics_runner.py),
  we use `llama-3.3-70b-versatile` as an AI Judge to score Faithfulness and
  Relevancy on a scale of 0.0 to 1.0.

---

### Concept 7: Hallucination Mitigation (How We Stop Fake Answers)

- **Simple Theory**: Hallucinations occur when an LLM makes up facts that were
  not in the provided documents.
- **In Our Project**: We stop hallucinations using a 4-layered defense:
  1. **Strict System Prompt**: Tells the model to use _only_ the provided
     context blocks and say "insufficient context" if the answer is missing.
  2. **Temperature 0.0**: Removes randomness from token generation.
  3. **Mandatory Citations**: Every claim must have a `[1]` or `[2]` bracketed
     citation.
  4. **Python Citation Guardrail**: Our code checks every citation number using
     regular expressions before returning the answer to the user.

---

_Interview Guide compiled for Enterprise Hybrid RAG Engine • Repository:
`divyyadav007/Advanced-NLP-Search-Retrieval-Engine`_
