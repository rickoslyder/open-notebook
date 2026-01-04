"""
Claude Agent SDK Chat Agent for Open Notebook.

This module provides an agentic chat interface that uses Claude Agent SDK
for intelligent RAG-powered conversations with notebook content.
"""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from open_notebook.agents.provider import (
    AgentProvider,
    ProviderConfig,
    get_agent_provider,
    get_sdk_environment,
    is_claude_agent_sdk_available,
)
from open_notebook.agents.tools import NOTEBOOK_TOOLS, handle_tool_call


# System prompt for the notebook chat agent
SYSTEM_PROMPT = """You are a helpful research assistant for Open Notebook, an AI-powered personal knowledge management system.

Your role is to help users explore, understand, and synthesize information from their notebooks, sources, and notes.

## Available Tools

You have access to tools for searching and retrieving notebook content:
- **search_notebook**: Search across sources and notes for relevant information
- **get_source**: Retrieve full content of a source document
- **get_note**: Retrieve full content of a note
- **list_notebook_sources**: List all sources in a notebook
- **list_notebook_notes**: List all notes in a notebook

## Guidelines

1. **Always search first**: When answering questions, use search_notebook to find relevant content before responding
2. **Cite your sources**: Always reference the sources/notes you used with [Source Title] or [Note Title] format
3. **Be comprehensive**: Synthesize information from multiple sources when relevant
4. **Be accurate**: Only state what is supported by the notebook content
5. **Ask for clarification**: If a question is ambiguous, ask for clarification
6. **Acknowledge limitations**: If information is not available in the notebook, say so clearly

## Response Format

When providing information from the notebook:
- Use clear markdown formatting
- Include citations in [brackets]
- Organize information logically
- Highlight key insights

Remember: You are helping users understand THEIR research and notes, so be helpful and thorough."""


@dataclass
class ChatMessage:
    """A chat message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatResult:
    """Result of a chat interaction."""
    response: str
    messages: List[ChatMessage]
    tool_calls_made: int = 0
    cost_usd: float = 0.0
    provider: str = "unknown"
    error: Optional[str] = None


class NotebookChatAgent:
    """
    Chat agent for Open Notebook powered by Claude Agent SDK.

    Supports fallback chain:
    1. Claude Max (via CLAUDE_CODE_OAUTH_TOKEN)
    2. Z.ai API (via ZAI_API_KEY)
    3. Legacy LangChain (via esperanto)
    """

    def __init__(
        self,
        notebook_id: str,
        session_id: Optional[str] = None,
        model_override: Optional[str] = None
    ):
        """
        Initialize the chat agent.

        Args:
            notebook_id: ID of the notebook for context
            session_id: Optional session ID for conversation continuity
            model_override: Optional model override
        """
        self.notebook_id = notebook_id
        self.session_id = session_id
        self.model_override = model_override
        self.messages: List[ChatMessage] = []
        self.provider: Optional[AgentProvider] = None
        self.provider_config: Optional[ProviderConfig] = None
        self._sdk_process: Optional[asyncio.subprocess.Process] = None

    async def _init_provider(self):
        """Initialize the best available provider."""
        self.provider, self.provider_config = get_agent_provider()
        logger.info(f"Chat agent initialized with provider: {self.provider.value}")

    async def chat(self, message: str) -> ChatResult:
        """
        Send a message and get a response.

        This method handles the full agent loop including tool calls.

        Args:
            message: User message

        Returns:
            ChatResult with response and metadata
        """
        if not self.provider:
            await self._init_provider()

        # Add user message to history
        self.messages.append(ChatMessage(role="user", content=message))

        if self.provider == AgentProvider.LEGACY:
            # Use legacy LangChain/LangGraph implementation
            return await self._chat_legacy(message)
        else:
            # Use Claude Agent SDK
            return await self._chat_sdk(message)

    async def _chat_sdk(self, message: str) -> ChatResult:
        """
        Chat using Claude Agent SDK.

        Uses subprocess to run Claude Agent SDK with our custom tools.
        """
        try:
            # Build the SDK query with context
            enriched_prompt = self._build_prompt(message)

            # For now, we use a simplified approach:
            # Call Claude API directly with tools, handling the agent loop ourselves
            # This avoids Node.js dependency while still getting agentic behavior

            result = await self._run_agent_loop(enriched_prompt)
            return result

        except Exception as e:
            logger.error(f"SDK chat error: {e}")
            # Fall back to legacy on error
            logger.info("Falling back to legacy provider due to SDK error")
            return await self._chat_legacy(message)

    async def _run_agent_loop(self, prompt: str) -> ChatResult:
        """
        Run the agent loop using Claude CLI via subprocess.

        Uses the `claude` CLI with the OAuth token from environment.
        The CLI handles the agent loop internally.
        """
        # Build full prompt with system context
        full_prompt = f"""You are a research assistant for a notebook application. You have access to the following notebook:

Notebook ID: {self.notebook_id}

Your task is to help answer questions about the notebook content. When searching, remember to use the notebook_id above.

User request: {prompt}

Please provide a helpful and comprehensive response based on the notebook content."""

        try:
            # Set up environment with OAuth token
            env = os.environ.copy()
            # HOME must be set for Claude CLI to work properly
            env["HOME"] = "/tmp"
            if self.provider == AgentProvider.CLAUDE_MAX:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = self.provider_config.api_key or ""

            # Get path to MCP config
            import open_notebook.agents
            agents_dir = os.path.dirname(open_notebook.agents.__file__)
            mcp_config_path = os.path.join(agents_dir, "mcp_config.json")

            # Use claude CLI with --print flag for non-interactive output
            # and --output-format json for structured response
            # --dangerously-skip-permissions allows automated tool usage without prompts
            # Run as non-root user 'claudeuser' since --dangerously-skip-permissions
            # cannot be used with root privileges for security reasons
            logger.info(f"Starting Claude CLI with MCP config: {mcp_config_path}")

            # Build the claude command to run as claudeuser
            claude_cmd = (
                f'claude --print --output-format json --dangerously-skip-permissions '
                f'--mcp-config "{mcp_config_path}" -p "{full_prompt.replace(chr(34), chr(92)+chr(34))}"'
            )

            # Use su to run as claudeuser with proper environment
            env["HOME"] = "/home/claudeuser"
            process = await asyncio.create_subprocess_exec(
                "su", "-s", "/bin/bash", "-c", claude_cmd, "claudeuser",
                stdin=asyncio.subprocess.DEVNULL,  # Prevent stdin waiting
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd="/home/claudeuser"  # Use claudeuser's home directory
            )

            logger.info("Waiting for Claude CLI response...")
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=180.0  # Increased timeout
            )
            logger.info(f"Claude CLI returned with code {process.returncode}")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Claude CLI error: {error_msg}")
                raise RuntimeError(f"Claude CLI failed: {error_msg}")

            # Parse JSON output from claude CLI
            output = stdout.decode().strip()

            # Handle different output formats
            try:
                result = json.loads(output)
                # Extract response from JSON structure
                if isinstance(result, dict):
                    response_text = result.get("result", result.get("response", result.get("content", str(result))))
                    cost_usd = result.get("total_cost_usd", result.get("cost_usd", result.get("costUsd", 0.0)))
                    num_turns = result.get("num_turns", 1)
                else:
                    response_text = str(result)
                    cost_usd = 0.0
                    num_turns = 1
            except json.JSONDecodeError:
                # If not valid JSON, use raw output
                response_text = output
                cost_usd = 0.0
                num_turns = 1

            # Add assistant message to history
            self.messages.append(ChatMessage(
                role="assistant",
                content=response_text,
                tool_calls=[],
                tool_results=[]
            ))

            return ChatResult(
                response=response_text,
                messages=self.messages,
                tool_calls_made=num_turns - 1,  # Turns minus initial request
                cost_usd=cost_usd,
                provider=self.provider.value
            )

        except asyncio.TimeoutError:
            logger.error("Claude CLI execution timed out")
            return ChatResult(
                response="Request timed out while processing.",
                messages=self.messages,
                provider=self.provider.value,
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"Claude CLI execution error: {e}")
            raise

    async def _chat_legacy(self, message: str) -> ChatResult:
        """
        Chat using legacy LangChain/LangGraph implementation.

        Falls back to the original Open Notebook chat graph.
        """
        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.runnables import RunnableConfig

            from open_notebook.graphs.chat import graph as chat_graph

            # Get current state
            config = RunnableConfig(configurable={"thread_id": self.session_id or self.notebook_id})
            current_state = chat_graph.get_state(config)

            # Prepare state
            state_values = current_state.values if current_state else {}
            state_values["messages"] = state_values.get("messages", [])
            state_values["context"] = await self._build_context()
            state_values["model_override"] = self.model_override

            # Add user message
            state_values["messages"].append(HumanMessage(content=message))

            # Execute graph
            result = chat_graph.invoke(
                input=state_values,
                config=config
            )

            # Extract response
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                response_text = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                response_text = "No response generated"

            # Add to history
            self.messages.append(ChatMessage(role="assistant", content=response_text))

            return ChatResult(
                response=response_text,
                messages=self.messages,
                provider="legacy"
            )

        except Exception as e:
            logger.error(f"Legacy chat error: {e}")
            return ChatResult(
                response=f"Error: {str(e)}",
                messages=self.messages,
                provider="legacy",
                error=str(e)
            )

    def _build_prompt(self, message: str) -> str:
        """Build enriched prompt with notebook context."""
        context = f"[Active Notebook ID: {self.notebook_id}]\n\n"
        context += "You are chatting with the user about their notebook. "
        context += "Use the available tools to search and retrieve relevant content.\n\n"
        context += f"User: {message}"
        return context

    async def _build_context(self) -> Dict[str, Any]:
        """Build context dict for legacy provider."""
        # Minimal context for legacy - it will load what it needs
        return {
            "notebook_id": self.notebook_id,
            "sources": [],
            "notes": []
        }

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []


class AgentSessionManager:
    """
    Manages active agent sessions.

    Provides session persistence and cleanup.
    """

    def __init__(self):
        self._sessions: Dict[str, NotebookChatAgent] = {}

    def get_or_create(
        self,
        notebook_id: str,
        session_id: str,
        model_override: Optional[str] = None
    ) -> NotebookChatAgent:
        """Get existing session or create new one."""
        key = f"{notebook_id}:{session_id}"

        if key not in self._sessions:
            self._sessions[key] = NotebookChatAgent(
                notebook_id=notebook_id,
                session_id=session_id,
                model_override=model_override
            )

        return self._sessions[key]

    def get(self, notebook_id: str, session_id: str) -> Optional[NotebookChatAgent]:
        """Get existing session or None."""
        key = f"{notebook_id}:{session_id}"
        return self._sessions.get(key)

    def remove(self, notebook_id: str, session_id: str):
        """Remove a session."""
        key = f"{notebook_id}:{session_id}"
        if key in self._sessions:
            del self._sessions[key]

    def clear_all(self):
        """Clear all sessions."""
        self._sessions.clear()


# Global session manager
session_manager = AgentSessionManager()
