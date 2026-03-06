from __future__ import annotations

from typing import Any, Dict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.prompts import SYSTEM_MESSAGE
from src.tools import risk_tools


load_dotenv(override=True)


# -----------------------------------------------------------------------------
# LLM with bound tools
# -----------------------------------------------------------------------------

llm_with_tools = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
).bind_tools(risk_tools)


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------


def reasoning_node(state: MessagesState) -> Dict[str, Any]:
    """
    Core reasoning node.

    Takes the current conversation state (messages), prepends the system message,
    calls the tool-enabled LLM, and returns the new assistant message.
    """
    messages = [SYSTEM_MESSAGE] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tools_node = ToolNode(risk_tools)


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------


def build_app():
    """Build and compile the LangGraph multi-agent workflow."""
    workflow = StateGraph(MessagesState)

    # Register nodes
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("tools_node", tools_node)

    # Entry point
    workflow.set_entry_point("reasoning_node")

    # Conditional routing from reasoning_node → tools_node or END
    workflow.add_conditional_edges(
        "reasoning_node",
        tools_condition,
        {
            "tools": "tools_node",
            END: END,
        },
    )

    # After tools execute, return control to reasoning_node
    workflow.add_edge("tools_node", "reasoning_node")

    # Simple in-memory checkpoint so conversation context is preserved
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


# Pre-built application instance
app = build_app()

