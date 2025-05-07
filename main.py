import ast
from itertools import chain
import time
import requests
import csv
import os
from prompt_techniques import PromptTechniques

import pandas as pd
#examples for few shot
examples = [{"input": "def add(a, b): return a + b", "output": "Parameters: a first number, b first number Returns: sum of two numbers"}, {"input": "def greet(name): print(f\"Hello, {name}\")", "output": "Greets a user by name."},
            {"input":"def DFS_REC(adj, visited, curr, result: "
                     "visited[curr] = True"
                     "result.append(curr)"
                     "for i in range(len(adj)):"
                     "  if not visited[i] and adj[s][i]== 1:"
                     "     DFS_REC(adj, visited, curr, result)",
             "output": "Recursively visits all adjacent vertices that are not visited yet. Parameters: adj = adjaceny matrix of all adjacent vertices, visited = list of booleans tracking which vertices have been visited, curr = current vertex, result "}]
def save_to_csv(output_path, records):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "filename",
        "repo",
        "file_url",
        "func_name",
        "language",
        "input_code",
        "model",
        "prompt",
        "generated_doc",
        "origin_doc",
    ]
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)



if __name__ == "__main__":
    #load dataset from csv file
    df = pd.read_csv('dataset/sample.csv')
    print(f"{df.info()}\nNumber of rows in dataset:{len(df)}")
    file_path = 'dataset/sample.csv'
    print(f"Processing file path: {file_path}")
    #user input
    print("Available models:\n")
    print("llama3.2:1b, qwen2.5-coder:0.5b, deepseek-r1:1.5b")
    model_used = input("\nEnter model name: ")
    print("Available prompts:\n")
    num = input("Choose(1, 2, 3, 4, or 5): zero_shot_records, few_shot_records, chain_records, structured_records, one_shot_records :\n")
    type_prompts = PromptTechniques(model_used)
    zero_shot_records = []
    few_shot_records = []
    chain_records = []
    structured_records = []
    one_shot_records = []

    records_map = {
        "1": zero_shot_records,
        "2": few_shot_records,
        "3": chain_records,
        "4": structured_records,
        "5": one_shot_records,
    }
    record = records_map[num]
    start_time = time.time()
    #model response called here
    print("loading generated doc...")
    for repo, file_url, func_name, language, func, func_tokens, docstring in zip(
            df['repository_name'], df['func_code_url'], df['func_name'],
            df['language'], df['func_code_string'], df['func_code_tokens'],
            df['func_documentation_string']):
        task = f"You are an expert Python developer and technical writer. Your task is to write concise and detailed Python docstring for the following function. Follow google-style format: \n{func}"
        print(f"Processing: {func_name}")
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

            case "5":
                one_doc = type_prompts.one_shot_prompting(task,examples)

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
                    "repo": repo,
                    "file_url": file_url,
                    "func_name": func_name,
                    "language": language,
                    "input_code": func,
                    "model": model_used,
                    "prompt": task,
                    "generated_doc": zero_doc
                    "origin_doc": docstring
                })
            case "2":
                few_shot_records.append({
                    "filename": os.path.basename(file_path),
                    "repo": repo,
                    "file_url": file_url,
                    "func_name": func_name,
                    "language": language,
                    "input_code": func,
                    "model": model_used,
                    "prompt": task,
                    "generated_doc": few_doc
                    "origin_doc": docstring
                })
            case "3":
                chain_records.append({
                    "filename": os.path.basename(file_path),
                    "repo": repo,
                    "file_url": file_url,
                    "func_name": func_name,
                    "language": language,
                    "input_code": func,
                    "model": model_used,
                    "prompt": task,
                    "generated_doc": chain_doc,
                    "origin_doc": docstring
                })
            case "4":
                structured_records.append({
                    "filename": os.path.basename(file_path),
                    "repo": repo,
                    "file_url": file_url,
                    "func_name": func_name,
                    "language": language,
                    "input_code": func,
                    "model": model_used,
                    "prompt": task,
                    "generated_doc": structured_doc,
                    "origin_doc": docstring
                })
            case "5":
                one_shot_records.append({
                    "filename": os.path.basename(file_path),
                    "function_name": func_name,
                    "model": model_used,
                    "prompt": task,
                    "input_code": func,
                    "generated_doc": one_doc,
                    "origin_doc":docstring
                })


    save_to_csv("output/generated_docs_.csv", record)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n Saved dataset to output/generated_docs_.csv. Total runtime {elapsed_time} seconds")
