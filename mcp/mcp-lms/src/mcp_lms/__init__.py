"""Canonical package for the LMS MCP server."""

from mcp_lms.client import LMSClient
from mcp_lms.server import create_server, main

__all__ = ["LMSClient", "create_server", "main"]
