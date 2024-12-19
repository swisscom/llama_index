import os
from typing import Any, Optional, List, Union, Dict

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai.utils import resolve_tool_choice

from llama_index.core.tools import BaseTool
from llama_index.llms.openai_like import OpenAILike


class SwissAIPlatform(OpenAILike):
    """
    Swiss AI Platform LlamaIndex integration by Swisscom.

    Examples:
        `pip install llama-index-llms-swissaiplatform`

        ```python
        from llama_index.llms.swissaiplatform import SwissAIPlatform

        # Set up the Groq class with the required model and API key
        llm = SwissAIPlatform(model="meta/llama-3.1-70b-instruct", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```

    api_base:
      - https://ep1.ai-platform.swisscom.com/v1 (endpoint 1, Swisscom external) - default
      - https://ep2.ai-platform.swisscom.com/v1 (endpoint 2, Swisscom external)
    """

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            api_base: str = "https://ep1.ai-platform.swisscom.com/v1",
            is_chat_model: bool = True,
            is_function_calling_model: bool = True,
            **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("SWISSAIPLATFORM_API_KEY", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "SwissAIPlatform"

    def _prepare_chat_with_tools(
            self,
            tools: List["BaseTool"],
            user_msg: Optional[Union[str, ChatMessage]] = None,
            chat_history: Optional[List[ChatMessage]] = None,
            verbose: bool = False,
            allow_parallel_tool_calls: bool = False,
            tool_choice: Union[str, dict] = "auto",
            strict: Optional[bool] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict and call the tool. Overwrite handling of parameters in openai llm integration.
           Does not specify strict and additionalProperties param in function tools as Swiss AI Platform does not know those params.

           Inspired by: llama_index.llms.openai base.py"""

        tool_specs = [tool.metadata.to_openai_tool() for tool in tools]

        # do not set function.strict param
        # do not set function.parameters.additionalProperties param

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice) if tool_specs else None,
            **kwargs,
        }
