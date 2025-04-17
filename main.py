import ast
import requests
import json

def extract_functions_from_file(file_path):
    """
    Parses a Python file and returns a list of function code blocks as strings.
    """
    with open(file_path, "r") as f:
        source_code = f.read()

    parsed = ast.parse(source_code)
    functions = []

    for node in parsed.body:
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = max(getattr(node, "end_lineno", start_line + 1), start_line + 1)
            func_lines = source_code.splitlines()[start_line:end_line]
            functions.append("\n".join(func_lines))

    return functions


def generate_doc_with_ollama(code_snippet, model="llama3.2", max_chars=800):
    prompt = f"Generate a concise and helpful documentation string for the following Python function:\n\n{code_snippet}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=30  # safety net if something hangs
        )
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Ollama request failed: {e}"

    result = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                result += data.get("response", "")
                if data.get("done", False) or len(result) >= max_chars:
                    break
            except json.JSONDecodeError:
                continue

    return result.strip() or "[No response from model]"


if __name__ == "__main__":
    functions = extract_functions_from_file("test_inputs/sample.py")

    for i, func in enumerate(functions):
        print(f"\nFunction {i+1}:\n{func}\n{'='*40}")

        doc = generate_doc_with_ollama(func)
        print(f"\n🧠 Generated Doc:\n{doc}\n{'-'*40}")
