"""A module for creating messages in a conversation."""

from typing import Literal, NamedTuple


class Message(NamedTuple):
    """A message in a conversation."""
    role: Literal['system', 'assistant', 'user']
    content: str


def assistant_message(text: str) -> Message:
    """Create a message from the assistant."""
    assert text, "Assistant message must not be empty"
    return Message(role="assistant", content=text)


def user_message(text: str) -> Message:
    """Create a message from the user."""
    assert text, "User message must not be empty"
    return Message(role="user", content=text)


def system_message(text: str) -> Message:
    """Create a message from the system."""
    return Message(role="system", content=text)
