import os
import sys
from textwrap import dedent

import requests


CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_PROMPT = dedent(
    """\
    You are a helpful assistant. Reply with three interesting facts about honeybees.
    """
).strip()


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Set the OPENAI_API_KEY environment variable with your ChatGPT API key before running.",
            file=sys.stderr,
        )
        return 1

    prompt = os.getenv("CHATGPT_DEMO_PROMPT", DEFAULT_PROMPT)

    try:
        response = requests.post(
            CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise assistant.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request to ChatGPT API failed: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(f"Status: {exc.response.status_code}", file=sys.stderr)
            print(f"Body: {exc.response.text}", file=sys.stderr)
        return 2

    payload = response.json()
    try:
        message_content = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        print(f"Unexpected API response format: {exc}\nPayload: {payload}", file=sys.stderr)
        return 3

    print("Prompt:")
    print(prompt)
    print("\nResponse:")
    print(message_content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
