from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


def _get_vectorstore() -> Chroma:
    """Return a Chroma vector store backed by a local persisted directory."""
    persist_dir = os.environ.get("CHROMA_DIR", "database/chroma_db")
    collection = os.environ.get("CHROMA_COLLECTION", "regulatory_fact_checker")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


@tool("query_documents_tool")
def query_documents_tool(query: str) -> str:
    """
    Retrieve the top 3 relevant regulatory document chunks from the local Chroma store.

    Returns the top 3 chunks with their metadata and text.
    """
    vs = _get_vectorstore()
    docs: List[Document] = vs.similarity_search(query, k=3)

    if not docs:
        return "No relevant documents were found in the vector store."

    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source_file") or meta.get("source") or "unknown"
        section = meta.get("section_title", "unknown section")
        page = meta.get("page_number", meta.get("page", "unknown"))
        effective_date = meta.get("effective_date", "unknown")
        topic_summary = meta.get("topic_summary", "n/a")

        header = (
            f"Chunk {i}:\n"
            f"- Source file: {source}\n"
            f"- Section title: {section}\n"
            f"- Page number: {page}\n"
            f"- Effective date: {effective_date}\n"
            f"- Topic summary: {topic_summary}\n"
        )
        parts.append(f"{header}\nText:\n{d.page_content}\n")

    return "\n---\n".join(parts)

