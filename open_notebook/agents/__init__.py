"""
Claude Agent SDK integration for Open Notebook.

This module provides an agentic chat interface powered by Claude Agent SDK,
with fallback support for multiple providers:
1. Claude Max (via CLAUDE_CODE_OAUTH_TOKEN)
2. Z.ai GLM API (via ZAI_API_KEY)
3. Existing multi-provider support (via esperanto/LangChain)
"""

from open_notebook.agents.chat_agent import NotebookChatAgent
from open_notebook.agents.provider import AgentProvider, get_agent_provider

__all__ = [
    "NotebookChatAgent",
    "AgentProvider",
    "get_agent_provider",
]
