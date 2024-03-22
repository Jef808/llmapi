"""Anthropic API client."""

from typing import Optional, List
from anthropic import Anthropic
from .apiclient import APIClient
from .message import Message


class AnthropicClient(APIClient):
    """A client for sending messages to the Anthropics API."""

    model_alias = {
        "big": "claude-3-opus-20240229",
        "medium": "claude-3-sonnet-20240229",
        "small": "claude-3-haiku-2024037",
    }

    def __init__(self,
                 *,
                 instructions="You are a useful assistant",
                 model="big",
                 stop: Optional[List[str]] = None,
                 temperature=0.7):
        super().__init__(client=Anthropic(),
                         instructions=instructions,
                         model=self.model_alias[model],
                         stop=stop,
                         temperature=temperature)

    def send_messages(self, messages: List[Message], *, max_tokens=600):
        """Create a chat completion."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stop_sequences": self.stop,
            "system": self.instructions
        }

        return self.client.messages.create(**payload)
