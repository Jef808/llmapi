"""OpenAI API client."""

from typing import List, Literal, Optional, Union
from openai import OpenAI
from llmapi.base_client import APIClient
from llmapi.message import Message, system_message


class OpenAIClient(APIClient):
    """A client for sending messages to the OpenAI API."""

    model_alias = {
        "big": "gpt-4-turbo-preview",
        "medium": "gpt-3.5-turbo",
        "small": "gpt-3.5-turbo"
    }

    def __init__(self,
                 *,
                 instructions: Optional[str],
                 model: Optional[Literal['big', 'medium', 'small']],
                 stop: Optional[Union[str, List[str]]],
                 temperature=0.7):
        super().__init__(client=OpenAI(),
                         instructions='You are a useful assistant' if instructions is None else instructions,
                         model=self.model_alias['big'] if model is None else self.model_alias[model],
                         stop=[stop] if stop is not None and isinstance(stop, str) else stop[:4] if stop is not None and isinstance(stop, list) else None,
                         temperature=temperature)

    def send(self, messages: Union[Message, List[Message]], *, max_tokens=600):
        """Create a chat completion."""
        if isinstance(messages, Message):
            messages = [messages]

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [system_message(self.instructions)._asdict(), *messages],
            "max_tokens": max_tokens,
            "stop": self.stop
        }

        response = self.client.chat.completions.create(**payload)
        return {
            "content": response.choices[0].message.content,
            **{k: v for k, v in response.model_dump().items() if k != 'choices'}
        }
