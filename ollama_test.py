import requests
import json
def _ollama_responses(model_name, prompt):

    print(f"\nModel:{model_name}")
    try:
        response = requests.post(
            url="http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Explain the fibonacci function",
                "stream": True,
            },
        )
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to Ollama: {e}")
        return
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                print(data.get("response", ""), end="")
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                print("\n Skipped a malformed line")
                continue

    print("\n Done.\n")

# Call each model
_ollama_responses("llama3.2", "Explain the Fibonacci function.")
_ollama_responses("qwen2.5-coder", "Explain the function.")



# response = requests.post(
#     "http://localhost:11434/api/generate",
#     json={
#         "model": "llama3.2",
#         "prompt": "Explain the Fibonacci function.",
#         "stream": True,
#     },
#     stream=True
# )
# response = requests.post(
#     url= "http://localhost:11434/api/generate",
#     json={
#         "model":"qwen2.5-coder",
#         "prompt": "Explain the function",
#         "stream": True,
#
#     },
#     stream=True
# )
#
# print("\nResponse from Ollama:")
# for line in response.iter_lines():
#     if line:
#         try:
#             data = json.loads(line.decode("utf-8"))
#             print(data.get("response", ""), end="")  # Print the actual text response
#             if data.get("done", False):
#                 break  # Stop when generation is complete
#         except json.JSONDecodeError:
#             continue  # Skip malformed lines
