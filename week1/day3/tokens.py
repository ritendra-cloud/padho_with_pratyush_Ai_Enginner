import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"


def ask_ollama(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "num_predict": 500,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama is not reachable. Start it with `ollama serve` and try again."
        ) from error


# 3 prompts
prompt1 = "Hi!"
prompt2 = "Explain time travel in detail but under 100 words."
prompt3 = "Write a 1000 word essay on machine learning."

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    response = ask_ollama(prompt)

    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens
    finish_reason = "stop" if response.get("done") else "length"

    print(f"Prompt: {prompt}")
    print(
        f"prompt_tokens: {prompt_tokens} | "
        f"completion_tokens: {completion_tokens} | "
        f"total_tokens: {total_tokens} | "
        f"finish_reason: {finish_reason}"
    )
    print("#######################################")
