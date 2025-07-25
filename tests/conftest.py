"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path

import asyncio


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# Test Document

This is a test document for VIBE.

## Features
- Feature 1
- Feature 2
- Feature 3

## Technical Requirements
- Backend: Python FastAPI
- Frontend: React
- Database: PostgreSQL

## API Endpoints
- GET /users
- POST /users
- PUT /users/{id}
- DELETE /users/{id}
"""


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing."""
    return {
        'GEMINI_API_KEY': 'test-gemini-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    }