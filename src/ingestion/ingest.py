from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Sequence

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Structured metadata extracted for each chunk of a regulatory document."""

    source_file: str
    page_number: int
    section_title: str = Field(
        description=(
            "The specific article, section, or chapter title "
            "(e.g., 'Article 4', 'Section 2.1'). "
            "If not explicitly present, infer the best short title."
        )
    )
    effective_date: str = Field(
        description=(
            "The date the regulation takes effect, if mentioned in the chunk. "
            "If unknown, return 'unknown'."
        )
    )
    topic_summary: str = Field(
        description=(
            "A 3-word summary of what this specific chunk is about. "
            "Exactly three words, lowercased, comma-free."
        )
    )


@dataclass
class IngestionConfig:
    input_dir: str = "data"
    persist_dir: str = "chroma_db"
    collection_name: str = "regulatory_fact_checker"
    chunk_size: int = 800
    chunk_overlap: int = 100
    # Default Gemini models targeting free-tier friendly endpoints; override via CLI if needed.
    embedding_model: str = "models/gemini-embedding-001"
    chat_model: str = "gemini-2.5-flash-lite"


def load_pdfs(input_dir: str) -> List[Document]:
    """Load all PDF documents from a directory using PyPDFLoader."""
    pdf_pattern = os.path.join(input_dir, "*.pdf")
    paths = sorted(glob.glob(pdf_pattern))

    if not paths:
        raise FileNotFoundError(f"No PDF files found in directory: {input_dir!r}")

    docs: List[Document] = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    return docs


def split_documents(
    documents: Sequence[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


def build_metadata_chain(chat_model: str):
    """Create an LLM chain that returns structured DocumentMetadata for each chunk."""
    llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0)
    structured_llm = llm.with_structured_output(DocumentMetadata)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert regulatory analyst. "
                    "Given a chunk of a regulatory document, extract precise metadata "
                    "according to the provided schema. Be conservative: do not invent "
                    "dates or titles that are not strongly implied by the text."
                ),
            ),
            (
                "human",
                (
                    "Source file: {source_file}\n"
                    "Page number: {page_number}\n\n"
                    "Chunk text:\n"
                    "--------------------\n"
                    "{chunk_text}\n"
                    "--------------------"
                ),
            ),
        ]
    )

    return prompt | structured_llm


def enrich_chunks_with_metadata(
    chunks: Sequence[Document],
    chain,
) -> List[Document]:
    """Call the metadata LLM for every chunk and attach the structured metadata."""
    enriched: List[Document] = []

    for idx, doc in enumerate(chunks, start=1):
        base_meta = dict(doc.metadata or {})
        source_file = base_meta.get("source") or base_meta.get("file_path") or "unknown"
        page_number = int(base_meta.get("page", base_meta.get("page_number", -1)))

        result: DocumentMetadata = chain.invoke(
            {
                "source_file": source_file,
                "page_number": page_number,
                "chunk_text": doc.page_content,
            }
        )

        # Pydantic v1/v2 compatibility
        if hasattr(result, "model_dump"):
            meta_dict = result.model_dump()
        else:
            meta_dict = result.dict()

        # Merge existing metadata with extracted metadata (extracted wins on conflicts).
        merged_meta = {**base_meta, **meta_dict}
        enriched.append(
            Document(page_content=doc.page_content, metadata=merged_meta)
        )

        # Lightweight progress indicator for long runs
        if idx % 25 == 0:
            print(f"[ingest] Processed metadata for {idx} chunks...")

    return enriched


def build_vectorstore(
    documents: Sequence[Document],
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
) -> Chroma:
    """Embed documents and create a persistent Chroma vector store."""
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    return vectorstore


def ingest(config: IngestionConfig) -> None:
    """Run the full ingestion pipeline end-to-end."""
    print(f"[ingest] Loading PDFs from: {config.input_dir}")
    raw_docs = load_pdfs(config.input_dir)
    print(f"[ingest] Loaded {len(raw_docs)} pages.")

    print(
        f"[ingest] Splitting documents "
        f"(chunk_size={config.chunk_size}, overlap={config.chunk_overlap})..."
    )
    chunks = split_documents(
        raw_docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    print(f"[ingest] Created {len(chunks)} chunks.")

    print("[ingest] Building metadata extraction chain...")
    metadata_chain = build_metadata_chain(config.chat_model)

    print("[ingest] Extracting metadata for each chunk (this may take a while)...")
    enriched_chunks = enrich_chunks_with_metadata(chunks, metadata_chain)
    print("[ingest] Metadata extraction complete.")

    print(
        f"[ingest] Building vector store in '{config.persist_dir}' "
        f"(collection='{config.collection_name}')..."
    )
    build_vectorstore(
        documents=enriched_chunks,
        persist_dir=config.persist_dir,
        collection_name=config.collection_name,
        embedding_model=config.embedding_model,
    )
    print("[ingest] Vector store built and persisted successfully.")


def parse_args() -> IngestionConfig:
    """Parse CLI arguments into an IngestionConfig."""
    parser = argparse.ArgumentParser(
        description="Ingest regulatory PDFs, extract metadata, and build a vector store.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory containing input PDF files (default: data).",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory to persist the Chroma vector store (default: chroma_db).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="regulatory_fact_checker",
        help="Name of the Chroma collection (default: regulatory_fact_checker).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for RecursiveCharacterTextSplitter (default: 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for RecursiveCharacterTextSplitter (default: 100).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="models/gemini-embedding-001",
        help="Embedding model name for GoogleGenerativeAIEmbeddings.",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini chat model name for metadata extraction.",
    )

    args = parser.parse_args()

    return IngestionConfig(
        input_dir=args.input_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )


if __name__ == "__main__":
    # Load environment variables like GOOGLE_API_KEY from a .env file if present.
    load_dotenv(override=True)

    cfg = parse_args()
    ingest(cfg)

