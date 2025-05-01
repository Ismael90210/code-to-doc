import ast
from itertools import chain
import time
import requests
import json
import csv
import os
from prompt_techniques import PromptTechniques
#examples for few shot
examples = [{"input": "def add(a, b): return a + b", "output": "Parameters: a first number, b first number Returns: sum of two numbers"}, {"input": "def greet(name): print(f\"Hello, {name}\")", "output": "Greets a user by name."},
            {"input":"def DFS_REC(adj, visited, curr, result: "
                     "visited[curr] = True"
                     "result.append(curr)"
                     "for i in range(len(adj)):"
                     "  if not visited[i] and adj[s][i]== 1:"
                     "     DFS_REC(adj, visited, curr, result)",
             "output": "Recursively visits all adjacent vertices that are not visited yet. Parameters: adj = adjaceny matrix of all adjacent vertices, visited = list of booleans tracking which vertices have been visited, curr = current vertex, result "}]

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
    Structured prompting technique
    """
    prompt = (f"You are an expert Python developer and technical writer. Your task is to write concise and detailed Python "
              f"docstring for the following function. Follow google-style format.\n{code_snippet} ")
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
    print("Available prompts:\n")
    num = input("Choose(1, 2, 3, or 4): zero_shot_records, few_shot_records, chain_records, structured_records:\n")
    type_prompts = PromptTechniques(model_used)
    #dataset_records = []
    zero_shot_records = []
    few_shot_records = []
    chain_records = []
    structured_records = []
    records_map = {
        "1": zero_shot_records,
        "2": few_shot_records,
        "3": chain_records,
        "4": structured_records,
    }
    record = records_map[num]
    file_path = "raw_code/fibonacci.py"
    print(f"\nProccessing {file_path}")
    functions = extract_functions_from_file(file_path)
    start_time = time.time()
    for i, func in enumerate(functions):
        #print(f"\nFunction {i + 1}:\n{func}\n{'=' * 40}")
        task = f"You are an expert Python developer and technical writer. Your task is to write concise and detailed Python docstring for the following function. Follow google-style format: \n{func}"
        #model response called here
        match num:
            case "1":
                zero_doc = type_prompts.zero_shot_prompting(task)
            case "2":
                few_doc = type_prompts.few_shot_prompting(task, examples)
            case "3":
                problem = (
                    f"Q:What is the appropriate google format docstring for the following function?: def add(a, b): return a + b. A: The function add has two parameters 'a' and 'b'."
                    f"The function body returns a+b, meaning add returns the arithmetic operation of addition between two integer's."
                    f"Docstring: Parameters: a first number, b second number"
                    f"Returns sum of a and b.")
                chain_doc = type_prompts.chain_of_thought_prompting(problem, func)
            case "4":
                structured_doc = type_prompts.structured_prompting(func)
        #doc = generate_doc_with_ollama(func, model_used)
        #print(f"\n Generated Doc:\n{doc}\n{'-' * 40}")

        try:
            func_name = func.strip().split('\n')[0].split('def')[1].split('(')[0].strip()
        except:
            func_name = f"function_{i + 1}"

        # dataset_records.append({
        #     "filename": os.path.basename(file_path),
        #     "function_name": func_name,
        #     "model": model_used,
        #     "input_code": func,
        #     "generated_doc": doc
        # })
        #Output
        match num:
            case "1":
                zero_shot_records.append({
                    "filename": os.path.basename(file_path),
                    "function_name": func_name,
                    "model": model_used,
                    "prompt": record,
                    "input_code": func,
                    "generated_doc": zero_doc
                })
            case "2":
                few_shot_records.append({
                    "filename": os.path.basename(file_path),
                    "function_name": func_name,
                    "model": model_used,
                    "prompt": record,
                    "input_code": func,
                    "generated_doc": few_doc
                })
            case "3":
                chain_records.append({
                    "filename": os.path.basename(file_path),
                    "function_name": func_name,
                    "model": model_used,
                    "prompt": record,
                    "input_code": func,
                    "generated_doc": chain_doc
                })
            case "4":
                structured_records.append({
                    "filename": os.path.basename(file_path),
                    "function_name": func_name,
                    "model": model_used,
                    "prompt": record,
                    "input_code": func,
                    "generated_doc": structured_doc
                })

    # save_to_csv("output/generated_docs_.csv", dataset_records)
    # save_to_csv("output/generated_docs_.csv", zero_shot_records)
    # save_to_csv("output/generated_docs_.csv", few_shot_records)
    # save_to_csv("output/generated_docs_.csv", chain_records)
    # save_to_csv("output/generated_docs_.csv", structured_records)
    save_to_csv("output/generated_docs_.csv", record)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n Saved dataset to output/generated_docs_.csv. sTotal runtime {elapsed_time} seconds")
