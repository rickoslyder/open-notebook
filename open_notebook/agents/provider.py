"""
Provider configuration for Claude Agent SDK with fallback chain.

Fallback order:
1. Claude Max (CLAUDE_CODE_OAUTH_TOKEN) - Uses Claude Max subscription
2. Z.ai API (ZAI_API_KEY) - Uses Z.ai GLM-4.7 via Anthropic-compatible API
3. Legacy (esperanto) - Falls back to existing multi-provider LangChain setup
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger


class AgentProvider(Enum):
    """Available agent providers in fallback order."""
    CLAUDE_MAX = "claude_max"
    ZAI_API = "zai_api"
    LEGACY = "legacy"


@dataclass
class ProviderConfig:
    """Configuration for an agent provider."""
    provider: AgentProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    available: bool = False

    def __post_init__(self):
        """Check if provider is available based on environment."""
        if self.provider == AgentProvider.CLAUDE_MAX:
            self.api_key = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
            self.available = bool(self.api_key)
            # Claude Max uses default Anthropic endpoint
            self.base_url = None
            self.model = "claude-sonnet-4-20250514"

        elif self.provider == AgentProvider.ZAI_API:
            self.api_key = os.getenv("ZAI_API_KEY")
            self.available = bool(self.api_key)
            self.base_url = "https://api.z.ai/api/anthropic"
            # Z.ai uses GLM-4.7 mapped to Claude model names
            self.model = os.getenv("ZAI_MODEL", "claude-sonnet-4-20250514")

        elif self.provider == AgentProvider.LEGACY:
            # Legacy is always available as final fallback
            self.available = True
            self.model = "legacy"


def get_agent_provider() -> tuple[AgentProvider, ProviderConfig]:
    """
    Get the best available agent provider based on environment configuration.

    Returns:
        Tuple of (provider enum, provider config)
    """
    # Check providers in fallback order
    providers = [
        ProviderConfig(AgentProvider.CLAUDE_MAX),
        ProviderConfig(AgentProvider.ZAI_API),
        ProviderConfig(AgentProvider.LEGACY),
    ]

    for config in providers:
        if config.available:
            logger.info(f"Using agent provider: {config.provider.value}")
            if config.provider == AgentProvider.CLAUDE_MAX:
                logger.debug("Claude Max subscription detected via CLAUDE_CODE_OAUTH_TOKEN")
            elif config.provider == AgentProvider.ZAI_API:
                logger.debug(f"Z.ai API detected, using endpoint: {config.base_url}")
            elif config.provider == AgentProvider.LEGACY:
                logger.debug("Falling back to legacy esperanto/LangChain provider")
            return config.provider, config

    # Should never reach here since LEGACY is always available
    raise RuntimeError("No agent provider available")


def get_sdk_environment() -> dict[str, str]:
    """
    Get environment variables needed for Claude Agent SDK.

    Returns:
        Dict of environment variables to set for SDK
    """
    provider, config = get_agent_provider()

    env = {}

    if provider == AgentProvider.CLAUDE_MAX:
        # Claude Max uses OAuth token
        env["CLAUDE_CODE_OAUTH_TOKEN"] = config.api_key or ""

    elif provider == AgentProvider.ZAI_API:
        # Z.ai uses Anthropic-compatible API with custom base URL
        env["ANTHROPIC_API_KEY"] = config.api_key or ""
        env["ANTHROPIC_BASE_URL"] = config.base_url or ""
        # Map models for Z.ai
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = "GLM-4.7"
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = "GLM-4.7"
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = "GLM-4.5-Air"

    return env


def is_claude_agent_sdk_available() -> bool:
    """
    Check if Claude Agent SDK can be used (Claude Max or Z.ai available).

    Returns:
        True if SDK can be used, False if only legacy provider available
    """
    provider, _ = get_agent_provider()
    return provider != AgentProvider.LEGACY
