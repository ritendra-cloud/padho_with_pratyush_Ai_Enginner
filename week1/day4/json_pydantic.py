import json
import urllib.error
import urllib.request

from pydantic import BaseModel


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"



class Ticket(BaseModel):
    name: str
    email: str
    issue: str

schema = Ticket.model_json_schema()

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""

text = "Hello My name is Pratyush. Yesterday I broke up with my girlfriend sheetal I have an iphone which is not working at all. My address is delhi. My email is abc@gmail.com. My contact number is 82134"
prompt = f"""
This is a customer ticket. Please extract the personal information from this.
{text}
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
    "format": schema,
    "stream": False,
}

request = urllib.request.Request(
    OLLAMA_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=180) as response:
        answer = json.loads(response.read().decode("utf-8"))["message"]["content"]
except urllib.error.URLError as error:
    raise RuntimeError(
        "Ollama is not reachable. Start it with `ollama serve` and try again."
    ) from error

print(answer)


# isko padhte kaise hai
import json
data_file = json.loads(answer)
ticket = Ticket.model_validate(data_file)


# inko pass kr sakte hai aage!
print(ticket.name)
print(ticket.email)
print(ticket.issue)



#Homework

# take resume in pdf or word
# have hr give you a list of things like skill, experience, projects
# extract these from resume 
# match against the hr list
# generate a percentage of matching or not
