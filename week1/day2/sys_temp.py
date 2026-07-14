import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"


def ask_ollama(messages: list[dict[str, str]], temperature: float = 0) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama is not reachable. Start it with `ollama serve` and try again."
        ) from error

    return data["message"]["content"].strip()


prompt = "Suggest a name for my clothing company."

# SYSTEM
message_system = {
    "role": "system",
    "content": (
        "You are a brand manager who suggests company names. "
        "The name should be one word. Suggest only two name."
    ),
}

# message me role and content
message = {
    "role": "user",
    "content": prompt,
}

messages = [message_system, message]

# Temperature controls creativity. 0 is focused; higher values are more random.
answer = ask_ollama(messages, temperature=2.5)

print("#######################################")
print(answer)
