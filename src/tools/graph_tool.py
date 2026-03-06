from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def _get_neo4j_graph() -> Neo4jGraph:
    """Return a Neo4jGraph instance connected to the Neo4j DB."""
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    if not user or not password:
        raise RuntimeError("NEO4J_USER and NEO4J_PASSWORD must be set.")

    return Neo4jGraph(
        url=uri,
        username=user,
        password=password,
        database="61faabc3"
    )


@tool("query_relationships_tool")
def query_relationships_tool(question: str) -> str:
    """
    Query relationships from the Neo4j credit risk graph.
    The input should be a natural language question about relationships.
    """
    graph = _get_neo4j_graph()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=False,
        allow_dangerous_requests=True,
        return_direct=True,
    )

    # Using the modern .invoke() method
    response = chain.invoke({"query": question})
    return response.get("result", "No result found.")
