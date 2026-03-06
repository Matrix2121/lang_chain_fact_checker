from __future__ import annotations

from .sql_tool import query_financials_tool
from .graph_tool import query_relationships_tool
from .vector_tool import query_documents_tool


risk_tools = [
    query_financials_tool,
    query_relationships_tool,
    query_documents_tool,
]

