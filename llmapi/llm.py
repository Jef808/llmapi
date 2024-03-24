#!/usr/bin/env python

"""A simple command-line chatbot client for OpenAI and Anthropic models."""

from typing import Literal, Optional, List
from llmapi.anthropic_client import AnthropicClient
from llmapi.openai_client import OpenAIClient
from llmapi.message import user_message, assistant_message
import click

def init_client(provider: Literal['openai', 'anthropic'],
                model=Literal['big', 'medium', 'small'],
                instructions: Optional[str] = None,
                stop: Optional[List[str]] = None,
                temperature=0.7):
    """Initialize a chatbot client."""
    if provider == 'openai':
        return OpenAIClient(instructions=instructions,
                            model=model,
                            stop=stop,
                            temperature=temperature)
    if provider == 'anthropic':
        return AnthropicClient(instructions=instructions,
                               model=model,
                               stop=stop,
                               temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")

@click.command()
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), required=True, help='Provider name')
@click.option('--model', type=click.Choice(['big', 'medium', 'small']), required=True, help='Model size')
@click.option('--instructions', type=str, help='Instructions for the model')
@click.option('--stop', multiple=True, help='Stop words list')
@click.option('--temperature', type=float, default=0.7, help='Temperature value')
@click.option('--max-tokens', type=int, default=600, help='Maximum output tokens')
@click.argument('messages', nargs=-1)
def main(provider, model, instructions, stop, temperature, max_tokens, messages):
    messages = [assistant_message(msg)._asdict() if i % 2 else user_message(msg)._asdict() for i, msg in enumerate(messages)]
    client = init_client(provider, model, instructions, stop, temperature)
    response = client.send(messages, max_tokens=max_tokens)
    print(response['content'])

if __name__ == '__main__':
    main()
