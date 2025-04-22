import ast
import requests
import json
import csv
import os

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

def generate_doc_with_ollama(code_snippet, model, max_chars=800):
    """
    Sends a function to the Ollama API and returns a generated docstring.
    """
    prompt = f"Generate a concise and helpful documentation string for the following Python functions. Above your documentation, provide an all caps label stating where the documentation starts, name it GENERATED DOCUMENTATION. Provide the amount of time taken to complete the task at the end of the file:\n\n{code_snippet}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=30
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

def save_to_csv(output_path, records):
    """
    Saves a list of dictionaries to a CSV file.
    Each dictionary should have keys: function_name, model, input_code, generated_doc
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "function_name", "model", "input_code", "generated_doc"])
        writer.writeheader()
        writer.writerows(records)

def get_all_python_files(directory):
    """
    Recursively find all .py files in a directory.
    """
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

if __name__ == "__main__":
    #user input
    print("Available models:\n")
    print("llama3.2:1b, qwen2.5-coder:0.5b, deepseek-r1:1.5b")

    model_used = input("\nEnter model name: ")
    dataset_records = []

    file_path = "raw_code/fibonacci.py"
    print(f"\nProccessing {file_path}")
    functions = extract_functions_from_file(file_path)
    for i, func in enumerate(functions):
        print(f"\nFunction {i + 1}:\n{func}\n{'=' * 40}")

        doc = generate_doc_with_ollama(func, model_used)
        print(f"\n Generated Doc:\n{doc}\n{'-' * 40}")

        try:
            func_name = func.strip().split('\n')[0].split('def')[1].split('(')[0].strip()
        except:
            func_name = f"function_{i + 1}"

        dataset_records.append({
            "filename": os.path.basename(file_path),
            "function_name": func_name,
            "model": model_used,
            "input_code": func,
            "generated_doc": doc
        })

    save_to_csv("output/generated_docs_fibonacci.csv", dataset_records)
    print("\n Saved dataset to output/generated_docs_fibonacci.csv")
