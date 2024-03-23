#!/usr/bin/env python3

from typing import Literal, Optional, List
from llmapi.anthropic_client import AnthropicClient
from llmapi.openai_client import OpenAIClient
from llmapi.message import user_message, assistant_message


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--provider', choices=['openai', 'anthropic'], required=True, help='Provider name')
    parser.add_argument('--model', choices=['big', 'medium', 'small'], required=True, help='Model size')
    parser.add_argument('--instructions', type=str, help='Instructions for the model')
    parser.add_argument('--stop', nargs='*', help='Stop words list')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature value')
    parser.add_argument('--max-tokens', type=int, default=600, help='Maximum output tokens')
    parser.add_argument('messages', nargs='+', help='Messages to process')

    args = parser.parse_args()
    messages = [assistant_message(msg)._asdict() if i % 2 else user_message(msg)._asdict() for i, msg in enumerate(args.messages)]

    client = init_client(args.provider, args.model, args.instructions, args.stop, args.temperature)
    response = client.send(messages, max_tokens=args.max_tokens)

    print(response['content'])
