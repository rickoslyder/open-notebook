"""
Claude Agent SDK Chat Router for Open Notebook.

This router provides the /api/v2/chat endpoints powered by Claude Agent SDK
with intelligent RAG capabilities and provider fallback.

Endpoints:
- POST /api/v2/chat/execute - Execute a chat message
- GET /api/v2/chat/sessions/{session_id} - Get session history
- DELETE /api/v2/chat/sessions/{session_id} - Delete a session
- GET /api/v2/chat/provider - Get current provider info
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from open_notebook.agents import NotebookChatAgent, get_agent_provider
from open_notebook.agents.chat_agent import session_manager
from open_notebook.agents.provider import is_claude_agent_sdk_available
from open_notebook.domain.notebook import Notebook
from open_notebook.exceptions import NotFoundError

router = APIRouter(prefix="/v2/chat", tags=["Agent Chat"])


# Request/Response Models

class AgentChatRequest(BaseModel):
    """Request for agent chat execution."""
    notebook_id: str = Field(..., description="Notebook ID for context")
    session_id: str = Field(..., description="Session ID for conversation continuity")
    message: str = Field(..., description="User message")
    model_override: Optional[str] = Field(None, description="Optional model override")


class AgentChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class AgentChatResponse(BaseModel):
    """Response from agent chat execution."""
    session_id: str = Field(..., description="Session ID")
    response: str = Field(..., description="Assistant response")
    messages: List[AgentChatMessage] = Field(..., description="Full conversation history")
    tool_calls_made: int = Field(0, description="Number of tool calls made")
    cost_usd: float = Field(0.0, description="Estimated cost in USD")
    provider: str = Field(..., description="Provider used (claude_max/zai_api/legacy)")
    error: Optional[str] = Field(None, description="Error message if any")


class SessionHistoryResponse(BaseModel):
    """Response with session history."""
    session_id: str = Field(..., description="Session ID")
    notebook_id: str = Field(..., description="Notebook ID")
    messages: List[AgentChatMessage] = Field(..., description="Conversation history")
    provider: str = Field(..., description="Current provider")


class ProviderInfoResponse(BaseModel):
    """Information about current agent provider."""
    provider: str = Field(..., description="Active provider name")
    sdk_available: bool = Field(..., description="Whether Claude Agent SDK is available")
    model: str = Field(..., description="Model being used")
    fallback_chain: List[str] = Field(..., description="Available fallback providers")


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = Field(True)
    message: str = Field(...)


# Endpoints

@router.post("/execute", response_model=AgentChatResponse)
async def execute_agent_chat(request: AgentChatRequest):
    """
    Execute a chat message using Claude Agent SDK.

    This endpoint provides intelligent RAG-powered chat with automatic
    tool use for searching and retrieving notebook content.

    The agent will:
    1. Search notebook sources and notes for relevant information
    2. Retrieve full content when needed
    3. Synthesize a response with proper citations

    Provider fallback order:
    1. Claude Max (if CLAUDE_CODE_OAUTH_TOKEN set)
    2. Z.ai API (if ZAI_API_KEY set)
    3. Legacy LangChain (always available)
    """
    try:
        # Verify notebook exists
        notebook = await Notebook.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Get or create agent session
        agent = session_manager.get_or_create(
            notebook_id=request.notebook_id,
            session_id=request.session_id,
            model_override=request.model_override
        )

        # Execute chat
        logger.info(f"Executing agent chat for notebook {request.notebook_id}, session {request.session_id}")
        result = await agent.chat(request.message)

        # Build response
        messages = [
            AgentChatMessage(role=msg.role, content=msg.content)
            for msg in result.messages
        ]

        return AgentChatResponse(
            session_id=request.session_id,
            response=result.response,
            messages=messages,
            tool_calls_made=result.tool_calls_made,
            cost_usd=result.cost_usd,
            provider=result.provider,
            error=result.error
        )

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except Exception as e:
        logger.error(f"Error in agent chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    notebook_id: str = Query(..., description="Notebook ID")
):
    """
    Get conversation history for a session.
    """
    try:
        agent = session_manager.get(notebook_id, session_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = [
            AgentChatMessage(role=msg.role, content=msg.content)
            for msg in agent.messages
        ]

        provider, _ = get_agent_provider()

        return SessionHistoryResponse(
            session_id=session_id,
            notebook_id=notebook_id,
            messages=messages,
            provider=provider.value
        )

    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def delete_session(
    session_id: str,
    notebook_id: str = Query(..., description="Notebook ID")
):
    """
    Delete a chat session.
    """
    try:
        session_manager.remove(notebook_id, session_id)
        return SuccessResponse(success=True, message="Session deleted")
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/provider", response_model=ProviderInfoResponse)
async def get_provider_info():
    """
    Get information about the current agent provider.

    Useful for debugging and understanding which backend is being used.
    """
    try:
        provider, config = get_agent_provider()
        sdk_available = is_claude_agent_sdk_available()

        return ProviderInfoResponse(
            provider=provider.value,
            sdk_available=sdk_available,
            model=config.model,
            fallback_chain=["claude_max", "zai_api", "legacy"]
        )

    except Exception as e:
        logger.error(f"Error getting provider info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-all-sessions", response_model=SuccessResponse)
async def clear_all_sessions():
    """
    Clear all active chat sessions.

    Use with caution - this will delete all conversation history.
    """
    try:
        session_manager.clear_all()
        return SuccessResponse(success=True, message="All sessions cleared")
    except Exception as e:
        logger.error(f"Error clearing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
