"""This module provides a client for sending messages to an API."""

from typing import Optional, List, Literal
from llmapi.message import Message


class APIClient:
    """A base class for clients to send messages to an API."""
    def __init__(self,
                 *,
                 client,
                 instructions: str,
                 model: Literal['big', 'medium', 'small'],
                 stop: Optional[List[str]],
                 temperature=0.4):
        self.client = client
        self.instructions = instructions
        self.model = model
        self.stop = stop
        self.temperature = temperature

    def send_messages(self, messages: List[Message], *, max_tokens):
        """Send messages to the API and return the response."""
