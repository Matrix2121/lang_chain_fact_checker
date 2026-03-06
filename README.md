# Regulatory Fact-Checker (LangChain RAG)

This project is a modular Retrieval-Augmented Generation (RAG) system for regulatory documents, with an automated metadata extraction pipeline and a quality-control evaluation layer.

The pipeline is split into:

- **`ingest.py`**: Data ingestion, PDF loading, chunking, and per-chunk metadata extraction before embedding and vector store creation.
- **`rag.py` (planned)**: RAG querying and evaluation (to be implemented based on your requirements).

## Setup

1. **Create and activate a virtual environment** (recommended).
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure your LLM/embeddings credentials** (Gemini API):

- Set `GOOGLE_API_KEY` in your environment, or
- Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

## Ingestion Pipeline (`ingest.py`)

- Loads all PDFs from an input directory using `PyPDFLoader`.
- Splits documents with `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=100`).
- For **every chunk**, calls an LLM with `.with_structured_output()` to populate a `DocumentMetadata` Pydantic model.
- Attaches the structured metadata to each chunk **before** embedding.
- Stores embeddings in a persistent vector store (e.g., Chroma) for later retrieval by the query/RAG layer.

See `ingest.py` for details and CLI usage.

