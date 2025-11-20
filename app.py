import os
import requests


CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_PROMPT = "Tell me three interesting facts about honeybees."


def main() -> int:
    api_key = os.environ["OPENAI_API_KEY"]
    prompt = os.getenv("CHATGPT_DEMO_PROMPT", DEFAULT_PROMPT)

    payload = requests.post(
        CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        },
        timeout=30,
    ).json()

    print("Prompt:")
    print(prompt)
    print("\nResponse:")
    print(payload["choices"][0]["message"]["content"].strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
