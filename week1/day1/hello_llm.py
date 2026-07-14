import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"


def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
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

    return data["message"]["content"]


prompt = (
    "Explain Python **kwargs. Return only 3 short numbered lines. "
    "No markdown code block. Line 1: meaning. Line 2: why useful. "
    "Line 3: tiny example."
)
answer = ask_ollama(prompt)

print(answer)
