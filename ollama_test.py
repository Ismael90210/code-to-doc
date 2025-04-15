import requests
import json

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Explain the Fibonacci function.",
        "stream": True,
    },
    stream=True
)

print("\nResponse from Ollama:")
for line in response.iter_lines():
    if line:
        try:
            data = json.loads(line.decode("utf-8"))
            print(data.get("response", ""), end="")  # Print the actual text response
            if data.get("done", False):
                break  # Stop when generation is complete
        except json.JSONDecodeError:
            continue  # Skip malformed lines
