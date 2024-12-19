import os

import pytest

from llama_index.llms.swissaiplatform import SwissAIPlatform


@pytest.mark.skipif("SWISSAIPLATFORM_API_KEY" not in os.environ, reason="No Swiss AI Platform API key")
def test_completion():
    swiss_ai_platform = SwissAIPlatform(model="meta/llama-3.1-70b-instruct", temperature=0, max_tokens=2)
    resp = swiss_ai_platform.complete("hello")
    assert resp.text == "Hello"


@pytest.mark.skipif("SWISSAIPLATFORM_API_KEY" not in os.environ, reason="No Swiss AI Platform API key")
def test_stream_completion():
    swiss_ai_platform = SwissAIPlatform(model="meta/llama-3.1-70b-instruct", temperature=0, max_tokens=2)
    stream = swiss_ai_platform.stream_complete("hello")
    text = None
    for chunk in stream:
        text = chunk.text
    assert text == "Hello"

# TODO add test for _prepare_chat_with_tools
