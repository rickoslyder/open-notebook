"""
MCP Tools for Open Notebook RAG functionality.

These tools allow Claude Agent SDK to search and retrieve content
from notebooks, sources, and notes.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger


# Tool definitions for Claude Agent SDK
# These will be registered as custom MCP tools

NOTEBOOK_TOOLS = [
    {
        "name": "search_notebook",
        "description": """Search through a notebook's sources and notes for relevant information.
Use this to find content related to a user's question. Returns snippets with source attribution.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question to find relevant content"
                },
                "notebook_id": {
                    "type": "string",
                    "description": "The notebook ID to search within"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["text", "vector", "hybrid"],
                    "default": "hybrid",
                    "description": "Type of search: text (keyword), vector (semantic), or hybrid (both)"
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results to return"
                },
                "search_sources": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to search in source documents"
                },
                "search_notes": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to search in notes"
                }
            },
            "required": ["query", "notebook_id"]
        }
    },
    {
        "name": "get_source",
        "description": """Get the full content of a specific source document.
Use this when you need complete details from a source mentioned in search results.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "The source ID to retrieve"
                },
                "include_insights": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include AI-generated insights about the source"
                }
            },
            "required": ["source_id"]
        }
    },
    {
        "name": "get_note",
        "description": """Get the full content of a specific note.
Use this when you need complete details from a note mentioned in search results.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "The note ID to retrieve"
                }
            },
            "required": ["note_id"]
        }
    },
    {
        "name": "list_notebook_sources",
        "description": """List all sources in a notebook.
Use this to get an overview of available source documents.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_id": {
                    "type": "string",
                    "description": "The notebook ID to list sources for"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of sources to return"
                }
            },
            "required": ["notebook_id"]
        }
    },
    {
        "name": "list_notebook_notes",
        "description": """List all notes in a notebook.
Use this to get an overview of available notes.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_id": {
                    "type": "string",
                    "description": "The notebook ID to list notes for"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of notes to return"
                }
            },
            "required": ["notebook_id"]
        }
    }
]


async def handle_search_notebook(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle search_notebook tool call.

    Uses the existing search infrastructure from Open Notebook.
    For notebook-scoped search, we first get the notebook's content,
    then use global search and filter results to those in the notebook.
    """
    from open_notebook.domain.notebook import Notebook, text_search, vector_search

    query = args["query"]
    notebook_id = args["notebook_id"]
    search_type = args.get("search_type", "hybrid")
    limit = args.get("limit", 5)
    search_sources = args.get("search_sources", True)
    search_notes = args.get("search_notes", True)

    try:
        # Ensure notebook_id has the correct prefix
        if not notebook_id.startswith("notebook:"):
            notebook_id = f"notebook:{notebook_id}"

        # Get notebook to verify it exists
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            return {
                "content": [{"type": "text", "text": f"Notebook {notebook_id} not found"}],
                "is_error": True
            }

        # Get the IDs of sources and notes in this notebook for filtering
        notebook_source_ids = set()
        notebook_note_ids = set()

        if search_sources:
            sources = await notebook.get_sources()
            notebook_source_ids = {s.id for s in sources}

        if search_notes:
            notes = await notebook.get_notes()
            notebook_note_ids = {n.id for n in notes}

        results = []

        # Use global search functions and filter to notebook content
        try:
            if search_type in ["vector", "hybrid"]:
                # Try vector search first
                search_results = await vector_search(
                    keyword=query,
                    results=limit * 3,  # Get more results to filter
                    source=search_sources,
                    note=search_notes,
                    minimum_score=0.2
                )
            else:
                search_results = []
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text: {e}")
            search_results = []

        # If vector search didn't work or hybrid mode, also try text search
        if search_type in ["text", "hybrid"] or not search_results:
            try:
                text_results = await text_search(
                    keyword=query,
                    results=limit * 3,
                    source=search_sources,
                    note=search_notes
                )
                # Merge results (avoid duplicates)
                existing_ids = {r.get("id") for r in search_results}
                for r in (text_results or []):
                    if r.get("id") not in existing_ids:
                        search_results.append(r)
            except Exception as e:
                logger.warning(f"Text search failed: {e}")

        # Filter results to only include items from this notebook
        for result in (search_results or []):
            result_id = result.get("id", "")
            result_type = None

            if result_id.startswith("source:") and result_id in notebook_source_ids:
                result_type = "source"
            elif result_id.startswith("note:") and result_id in notebook_note_ids:
                result_type = "note"
            elif result_id.startswith("source_embedding:"):
                # Source embeddings - check if parent source is in notebook
                # The search result should have source info
                parent_source_id = result.get("source", {}).get("id") if isinstance(result.get("source"), dict) else result.get("source")
                if parent_source_id and str(parent_source_id) in notebook_source_ids:
                    result_type = "source"
                    result_id = str(parent_source_id)

            if result_type:
                content = result.get("content") or result.get("full_text") or ""
                title = result.get("title") or "Untitled"
                preview = content[:500] if content else ""

                results.append({
                    "type": result_type,
                    "id": result_id,
                    "title": title,
                    "preview": preview + "..." if len(content) > 500 else preview,
                    "score": result.get("score")
                })

                if len(results) >= limit:
                    break

        if not results:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No results found for query: '{query}' in notebook {notebook_id}"
                }]
            }

        # Format results with citations
        formatted = f"Found {len(results)} results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"[{i}] {r['type'].upper()}: {r['title']}\n"
            formatted += f"    ID: {r['id']}\n"
            formatted += f"    Preview: {r['preview']}\n\n"

        return {
            "content": [{"type": "text", "text": formatted}]
        }

    except Exception as e:
        logger.error(f"Error in search_notebook: {e}")
        return {
            "content": [{"type": "text", "text": f"Search error: {str(e)}"}],
            "is_error": True
        }


async def handle_get_source(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle get_source tool call.

    Retrieves full source content with optional insights.
    """
    from open_notebook.domain.notebook import Source

    source_id = args["source_id"]
    include_insights = args.get("include_insights", True)

    try:
        # Ensure source_id has the correct prefix
        if not source_id.startswith("source:"):
            source_id = f"source:{source_id}"

        source = await Source.get(source_id)
        if not source:
            return {
                "content": [{"type": "text", "text": f"Source {source_id} not found"}],
                "is_error": True
            }

        # Build response
        response = f"# {source.title or 'Untitled Source'}\n\n"
        response += f"**ID:** {source.id}\n"

        # Check for asset with URL
        if source.asset and source.asset.url:
            response += f"**URL:** {source.asset.url}\n"
        elif source.asset and source.asset.file_path:
            response += f"**File:** {source.asset.file_path}\n"

        # Use full_text field (the actual content field in Source model)
        content = source.full_text or "No content available"
        response += f"\n## Content\n\n{content}\n"

        if include_insights:
            try:
                insights = await source.get_insights()
                if insights:
                    response += "\n## AI Insights\n\n"
                    for insight in insights:
                        response += f"### {insight.insight_type}\n{insight.content}\n\n"
            except Exception as e:
                logger.warning(f"Could not fetch insights: {e}")

        return {
            "content": [{"type": "text", "text": response}]
        }

    except Exception as e:
        logger.error(f"Error in get_source: {e}")
        return {
            "content": [{"type": "text", "text": f"Error retrieving source: {str(e)}"}],
            "is_error": True
        }


async def handle_get_note(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle get_note tool call.

    Retrieves full note content.
    """
    from open_notebook.domain.notebook import Note

    note_id = args["note_id"]

    try:
        # Ensure note_id has the correct prefix
        if not note_id.startswith("note:"):
            note_id = f"note:{note_id}"

        note = await Note.get(note_id)
        if not note:
            return {
                "content": [{"type": "text", "text": f"Note {note_id} not found"}],
                "is_error": True
            }

        response = f"# {note.title or 'Untitled Note'}\n\n"
        response += f"**ID:** {note.id}\n"
        if note.note_type:
            response += f"**Type:** {note.note_type}\n"
        if note.created:
            response += f"**Created:** {note.created}\n"
        if note.updated:
            response += f"**Updated:** {note.updated}\n"
        response += f"\n## Content\n\n{note.content or 'No content available'}\n"

        return {
            "content": [{"type": "text", "text": response}]
        }

    except Exception as e:
        logger.error(f"Error in get_note: {e}")
        return {
            "content": [{"type": "text", "text": f"Error retrieving note: {str(e)}"}],
            "is_error": True
        }


async def handle_list_notebook_sources(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle list_notebook_sources tool call.

    Lists all sources in a notebook.
    """
    from open_notebook.domain.notebook import Notebook

    notebook_id = args["notebook_id"]
    limit = args.get("limit", 20)

    try:
        # Ensure notebook_id has the correct prefix
        if not notebook_id.startswith("notebook:"):
            notebook_id = f"notebook:{notebook_id}"

        notebook = await Notebook.get(notebook_id)
        if not notebook:
            return {
                "content": [{"type": "text", "text": f"Notebook {notebook_id} not found"}],
                "is_error": True
            }

        sources = await notebook.get_sources()

        if not sources:
            return {
                "content": [{"type": "text", "text": f"No sources found in notebook {notebook_id}"}]
            }

        # Apply limit
        sources = sources[:limit]

        response = f"# Sources in Notebook\n\nFound {len(sources)} sources:\n\n"
        for i, source in enumerate(sources, 1):
            response += f"{i}. **{source.title or 'Untitled'}** (ID: {source.id})\n"
            if source.asset and source.asset.url:
                response += f"   URL: {source.asset.url}\n"
            if source.topics:
                response += f"   Topics: {', '.join(source.topics)}\n"

        return {
            "content": [{"type": "text", "text": response}]
        }

    except Exception as e:
        logger.error(f"Error in list_notebook_sources: {e}")
        return {
            "content": [{"type": "text", "text": f"Error listing sources: {str(e)}"}],
            "is_error": True
        }


async def handle_list_notebook_notes(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle list_notebook_notes tool call.

    Lists all notes in a notebook.
    """
    from open_notebook.domain.notebook import Notebook

    notebook_id = args["notebook_id"]
    limit = args.get("limit", 20)

    try:
        # Ensure notebook_id has the correct prefix
        if not notebook_id.startswith("notebook:"):
            notebook_id = f"notebook:{notebook_id}"

        notebook = await Notebook.get(notebook_id)
        if not notebook:
            return {
                "content": [{"type": "text", "text": f"Notebook {notebook_id} not found"}],
                "is_error": True
            }

        notes = await notebook.get_notes()

        if not notes:
            return {
                "content": [{"type": "text", "text": f"No notes found in notebook {notebook_id}"}]
            }

        # Apply limit
        notes = notes[:limit]

        response = f"# Notes in Notebook\n\nFound {len(notes)} notes:\n\n"
        for i, note in enumerate(notes, 1):
            preview = (note.content[:100] + "...") if note.content and len(note.content) > 100 else (note.content or "")
            response += f"{i}. **{note.title or 'Untitled'}** (ID: {note.id})\n"
            if note.note_type:
                response += f"   Type: {note.note_type}\n"
            response += f"   Preview: {preview}\n"

        return {
            "content": [{"type": "text", "text": response}]
        }

    except Exception as e:
        logger.error(f"Error in list_notebook_notes: {e}")
        return {
            "content": [{"type": "text", "text": f"Error listing notes: {str(e)}"}],
            "is_error": True
        }


# Tool handler registry
TOOL_HANDLERS = {
    "search_notebook": handle_search_notebook,
    "get_source": handle_get_source,
    "get_note": handle_get_note,
    "list_notebook_sources": handle_list_notebook_sources,
    "list_notebook_notes": handle_list_notebook_notes,
}


async def handle_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route a tool call to the appropriate handler.

    Args:
        tool_name: Name of the tool to call
        args: Arguments for the tool

    Returns:
        Tool result dict with content and optional is_error flag
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "is_error": True
        }

    return await handler(args)
