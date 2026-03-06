from __future__ import annotations

import json
import os
import uuid
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.agent.orchestrator import app as graph_app


# Load environment variables early so LangSmith settings are picked up.
load_dotenv(override=True)


st.set_page_config(page_title="Corporate Risk Assessor", page_icon="📊")
st.title("Corporate Risk Assessor")


# -----------------------------------------------------------------------------
# LangSmith tracing status (sidebar)
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Observability")
    if os.environ.get("LANGCHAIN_API_KEY"):
        st.success("🟢 LangSmith Tracing Active", icon="✅")
    else:
        st.warning(
            "🟡 LangSmith Tracing Disabled (Add API Key to .env)",
            icon="⚠️",
        )


# -----------------------------------------------------------------------------
# Session state setup
# -----------------------------------------------------------------------------


if "messages" not in st.session_state:
    # Each item: {"role": "user" | "assistant", "content": str}
    st.session_state.messages: List[Dict[str, str]] = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


render_chat_history()


# -----------------------------------------------------------------------------
# Chat input & LangGraph interaction
# -----------------------------------------------------------------------------


user_input = st.chat_input(
    "Ask about a company's risk profile, exposures, or relationships..."
)

if user_input:
    # Add user message to local UI history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare input for LangGraph app (MessagesState)
    graph_input = {
        "messages": [HumanMessage(content=user_input)],
    }

    assistant_answer: str | None = None

    with st.status("Assessing Risk Profile...", expanded=True) as status:
        # 1. Use "updates" instead of "events" to get clean state outputs
        for event in graph_app.stream(
            graph_input,
            config={"configurable": {"thread_id": st.session_state.thread_id}},
            stream_mode="updates",  # <--- THIS IS THE MAGIC FIX
        ):
            # 'event' will look like: {"tools_node": {...}} OR {"reasoning_node": {...}}
            for node_name, node_state in event.items():
                
                # Update UI when tools are running
                if node_name == "tools_node":
                    status.update(
                        label="Assessing Risk Profile... (Analyzing data with tools)",
                        state="running",
                    )
                
                # Capture the AI's response when the reasoning node finishes
                elif node_name == "reasoning_node":
                    messages = node_state.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        
                        # Guard: Ensure it's an AI message AND it's not trying to call a tool
                        is_ai = getattr(last_msg, "type", "") in ("ai", "AIMessage")
                        has_tool_calls = getattr(last_msg, "tool_calls", False)
                        
                        if is_ai and not has_tool_calls:
                            content = getattr(last_msg, "content", "")
                            
                            # Handle standard plain strings
                            if isinstance(content, str) and content.strip():
                                assistant_answer = content.strip()
                            
                            # Handle Gemini's list of content blocks
                            elif isinstance(content, list):
                                extracted = [
                                    b["text"] for b in content 
                                    if isinstance(b, dict) and "text" in b
                                ]
                                combined = "".join(extracted).strip()
                                if combined:
                                    assistant_answer = combined

        # After streaming is done, finalize status
        if assistant_answer:
            status.update(
                label="Risk assessment complete.",
                state="complete",
            )
        else:
            status.update(
                label="Risk assessment finished, but no response generated.",
                state="error",
            )

    # Display assistant reply in chat and persist to session
    if assistant_answer:
        with st.chat_message("assistant"):
            st.markdown(assistant_answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_answer}
        )
