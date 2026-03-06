from __future__ import annotations

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

import os

load_dotenv()


def _get_sql_db() -> SQLDatabase:
    """Return a SQLDatabase connection to the local SQLite risk_data.db."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # Point to the database folder
    db_path = os.path.join(root_dir, "database", "risk_data.db").replace("\\", "/")
    
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")


def _get_sql_llm() -> ChatGoogleGenerativeAI:
    """LLM used for NL -> SQL translation."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _extract_select_sql(raw_content: str) -> str:
    """
    Extract a clean SELECT statement from LLM output.

    Handles cases where the model wraps SQL in markdown fences (```sql ... ```)
    or adds explanatory text before the query.
    """
    text = (raw_content or "").strip()

    # If there's a fenced code block, extract its contents.
    if "```" in text:
        first = text.find("```")
        second = text.find("```", first + 3)
        if second != -1:
            code_block = text[first + 3 : second]
            lines = code_block.splitlines()
            # Drop language tag like "sql" on the first line if present.
            if lines and lines[0].strip().lower().startswith("sql"):
                lines = lines[1:]
            text = "\n".join(lines).strip()

    # If there is leading explanation text, cut from the first SELECT onward.
    upper = text.upper()
    idx = upper.find("SELECT")
    if idx != -1:
        text = text[idx:]

    return text.strip()


@tool("query_financials_tool")
def query_financials_tool(question: str) -> str:
    """
    Query structured financial data from the local SQLite credit risk database.

    The input should be a natural language question about clients, loans, or risk.
    The tool translates the question into a safe, read-only SQL SELECT query,
    executes it, and returns the results as text.
    """
    db = _get_sql_db()
    llm = _get_sql_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a SQL assistant for a credit risk application. "
                    "You have access to a SQLite database with the following tables:\n"
                    "- Clients(ClientID, Name, Turnover, RiskRating)\n"
                    "- Loans(LoanID, ClientID, Amount, InterestRate, Status)\n\n"
                    "Given a natural language question, write a SINGLE SQL query "
                    "that answers it. VERY IMPORTANT:\n"
                    "- The query MUST be a read-only SELECT statement.\n"
                    "- Do NOT use INSERT, UPDATE, DELETE, DROP, or any DDL/DML.\n"
                    "- Do NOT modify the database in any way.\n"
                    "- Do NOT include comments or explanations, only the SQL.\n"
                ),
            ),
            ("human",
             "Question: {question}\n\nReturn ONLY the SQL SELECT query."),
        ]
    )

    raw_content = llm.invoke(prompt.format(question=question)).content
    sql_query = _extract_select_sql(raw_content)

    normalized = sql_query.lstrip().upper() if sql_query else ""
    if not normalized.startswith("SELECT"):
        return (
            "The model did not produce a safe read-only SELECT query. "
            "Refusing to execute. Generated text was:\n"
            f"{raw_content}"
        )

    try:
        result = db.run(sql_query)
    except Exception as e:
        return f"Error executing SQL query: {e}\nQuery was:\n{sql_query}"

    return f"SQL query:\n{sql_query}\n\nResult:\n{result}"
