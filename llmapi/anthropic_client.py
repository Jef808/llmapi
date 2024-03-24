"""Anthropic API client."""

from typing import List, Literal, Optional, Union
from anthropic import Anthropic
from llmapi.base_client import APIClient
from llmapi.message import Message


class AnthropicClient(APIClient):
    """A client for sending messages to the Anthropics API."""

    model_alias = {
        "big": "claude-3-opus-20240229",
        "medium": "claude-3-sonnet-20240229",
        "small": "claude-3-haiku-2024037",
    }

    def __init__(self,
                 *,
                 instructions: Optional[str],
                 model=Literal['big', 'medium', 'small'],
                 stop: Optional[Union[str, List[str]]],
                 temperature=0.7):
        super().__init__(client=Anthropic(),
                         instructions="You are a useful assistant" if instructions is None else instructions,
                         model=self.model_alias['big'] if model is None else self.model_alias[model],
                         stop=[stop] if stop is not None and isinstance(stop, str) else stop,
                         temperature=temperature)

    def send(self, messages: Union[Message, List[Message]], *, max_tokens=600):
        """Create a chat completion."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stop_sequences": self.stop,
            "system": self.instructions
        }

        response = self.client.messages.create(**payload)
        return {
            "content": response.content[0].text,
            **{k: v for k, v in response.model_dump().items() if k != 'content'}
        }
