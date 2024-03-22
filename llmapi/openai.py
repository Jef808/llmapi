"""OpenAI API client."""

from typing import List, Optional
from openai import OpenAI, OpenAIError
from .apiclient import APIClient
from .message import Message, system_message


class OpenAIClient(APIClient):
    """A client for sending messages to the OpenAI API."""

    model_alias = {
        "big": "gpt-4-turbo-preview",
        "medium": "gpt-3.5-turbo"
    }

    def __init__(self,
                 *,
                 instructions="You are a useful assistant",
                 model="big",
                 stop: Optional[List[str]] = None,
                 temperature=0.7):
        super().__init__(client=OpenAI(),
                         instructions=instructions,
                         model=self.model_alias[model],
                         stop=stop[:4] if stop is not None else None,
                         temperature=temperature)

    def send_messages(self, messages: List[Message], *, max_tokens=600):
        """Create a chat completion."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [system_message(self.instructions), *messages],
            "max_tokens": max_tokens,
            "stop": self.stop
        }

        return self.client.chat.completions.create(**payload)
