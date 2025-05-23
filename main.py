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
             "output": "Recursively visits all adjacent vertices that are not visited yet. Parameters: adj = adjaceny matrix of all adjacent vertices, visited = list of booleans tracking which vertices have been visited, curr = current vertex, result "},
            {
                "input": "def square(n): return n ** 2",
                "output": "Calculates the square of a number.\n\nArgs:\n    n (int or float): The number to square.\n\nReturns:\n    int or float: The squared result."
            },
            {
                "input": "def is_even(x): return x % 2 == 0",
                "output": "Checks if a number is even.\n\nArgs:\n    x (int): The number to check.\n\nReturns:\n    bool: True if even, False otherwise."
            },
            {
                "input": "def reverse_string(s): return s[::-1]",
                "output": "Reverses a string.\n\nArgs:\n    s (str): The string to reverse.\n\nReturns:\n    str: The reversed string."
            },
            {
                "input": "def max_of_two(a, b): return a if a > b else b",
                "output": "Finds the maximum of two values.\n\nArgs:\n    a (int or float): First number.\n    b (int or float): Second number.\n\nReturns:\n    int or float: The larger of the two values."
            },
            {
                "input": "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')",
                "output": "Counts the number of vowels in a string.\n\nArgs:\n    s (str): The input string.\n\nReturns:\n    int: The number of vowels."
            },
            {
                "input": "def factorial(n): return 1 if n == 0 else n * factorial(n - 1)",
                "output": "Calculates the factorial of a number recursively.\n\nArgs:\n    n (int): A non-negative integer.\n\nReturns:\n    int: The factorial of n."
            },
            {
                "input": "def remove_duplicates(lst): return list(set(lst))",
                "output": "Removes duplicate elements from a list.\n\nArgs:\n    lst (list): A list that may contain duplicates.\n\nReturns:\n    list: A list with unique elements only."
            },
            {
                "input": "def average(numbers): return sum(numbers) / len(numbers) if numbers else 0",
                "output": "Calculates the average of a list of numbers.\n\nArgs:\n    numbers (list): List of numeric values.\n\nReturns:\n    float: The average of the values, or 0 if the list is empty."
            },
            {
                "input": "def merge_lists(a, b): return a + b",
                "output": "Merges two lists into one.\n\nArgs:\n    a (list): The first list.\n    b (list): The second list.\n\nReturns:\n    list: A new list containing all elements from a and b."
            },
            {
                "input": "def is_palindrome(s): return s == s[::-1]",
                "output": "Checks if a string is a palindrome.\n\nArgs:\n    s (str): The string to check.\n\nReturns:\n    bool: True if the string is a palindrome, False otherwise."
            },
            {
                "input": "def sum_list(lst): return sum(lst)",
                "output": "Calculates the sum of all elements in a list.\n\nArgs:\n    lst (list): A list of numeric values.\n\nReturns:\n    int or float: The sum of the list elements."
            },
            {
                "input": "def get_unique_elements(lst): return list(set(lst))",
                "output": "Returns unique elements from a list.\n\nArgs:\n    lst (list): Input list.\n\nReturns:\n    list: List of unique elements."
            },
            {
                "input": "def capitalize_words(s): return ' '.join(word.capitalize() for word in s.split())",
                "output": "Capitalizes the first letter of each word in a string.\n\nArgs:\n    s (str): Input string.\n\nReturns:\n    str: String with capitalized words."
            },
            {
                "input": "def multiply_list(lst): result = 1; for x in lst: result *= x; return result",
                "output": "Calculates the product of all elements in a list.\n\nArgs:\n    lst (list): A list of numbers.\n\nReturns:\n    int or float: The product of the list elements."
            },
            {
                "input": "def flatten_list(nested_list): return [item for sublist in nested_list for item in sublist]",
                "output": "Flattens a list of lists into a single list.\n\nArgs:\n    nested_list (list): List of lists.\n\nReturns:\n    list: Flattened list."
            },
            {
                "input": "def get_file_extension(filename): return filename.split('.')[-1]",
                "output": "Gets the file extension from a filename.\n\nArgs:\n    filename (str): Name of the file.\n\nReturns:\n    str: File extension."
            },
            {
                "input": "def is_substring(sub, s): return sub in s",
                "output": "Checks if a substring exists within a string.\n\nArgs:\n    sub (str): Substring.\n    s (str): String to search.\n\nReturns:\n    bool: True if substring is found, False otherwise."
            },
            {
                "input": "def find_max(lst): return max(lst)",
                "output": "Finds the maximum value in a list.\n\nArgs:\n    lst (list): List of numbers.\n\nReturns:\n    int or float: Maximum value."
            },
            {
                "input": "def count_occurrences(lst, x): return lst.count(x)",
                "output": "Counts how many times x appears in a list.\n\nArgs:\n    lst (list): List of elements.\n    x: Element to count.\n\nReturns:\n    int: Number of occurrences."
            },
            {
                "input": "def get_keys(d): return list(d.keys())",
                "output": "Returns the keys of a dictionary.\n\nArgs:\n    d (dict): Input dictionary.\n\nReturns:\n    list: List of keys."
            },
            {
                "input": "def get_values(d): return list(d.values())",
                "output": "Returns the values of a dictionary.\n\nArgs:\n    d (dict): Input dictionary.\n\nReturns:\n    list: List of values."
            },
            {
                "input": "def reverse_list(lst): return lst[::-1]",
                "output": "Reverses a list.\n\nArgs:\n    lst (list): Input list.\n\nReturns:\n    list: Reversed list."
            },
            {
                "input": "def square_list(lst): return [x**2 for x in lst]",
                "output": "Squares each element in a list.\n\nArgs:\n    lst (list): List of numbers.\n\nReturns:\n    list: List of squared numbers."
            },
            {
                "input": "def zip_lists(a, b): return list(zip(a, b))",
                "output": "Zips two lists together into pairs.\n\nArgs:\n    a (list): First list.\n    b (list): Second list.\n\nReturns:\n    list: List of tuples pairing elements from a and b."
            },
            {
                "input": "def filter_even(lst): return [x for x in lst if x % 2 == 0]",
                "output": "Filters even numbers from a list.\n\nArgs:\n    lst (list): List of numbers.\n\nReturns:\n    list: List of even numbers."
            },
            {
                "input": "def filter_odd(lst): return [x for x in lst if x % 2 != 0]",
                "output": "Filters odd numbers from a list.\n\nArgs:\n    lst (list): List of numbers.\n\nReturns:\n    list: List of odd numbers."
            },
            {
                "input": "def join_strings(lst): return ' '.join(lst)",
                "output": "Joins a list of strings into a single string with spaces.\n\nArgs:\n    lst (list): List of strings.\n\nReturns:\n    str: Combined string."}
            ]
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
    df = pd.read_csv('dataset/tadeck_onetimepass.csv')
    print(f"{df.info()}\nNumber of rows in dataset:{len(df)}")
    file_path = 'dataset/tadeck_onetimepass.csv'
    print(f"Processing file path: {file_path}")
    #model selection
    print("Available models:\n")
    print("llama3.2:1b, qwen2.5-coder:0.5b, deepseek-r1:1.5b")
    model_used = input("\nEnter model name: ")
    #prompt technique
    print("Available prompts:\n")
    num = input("Choose(1, 2, 3, 4, or 5): zero_shot_records, few_shot_records, chain_records, structured_records, one_shot_records :\n")
    type_prompts = PromptTechniques(model_used)
    num_prompts = 2
    #int(input("Choose the number of prompts: ")))
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
    prompt = records_map[num]
    output_name = ""
    start_time = time.time()
    #model response called here
    print("loading generated doc...")
    #extracting columns from dataset
    for repo, file_url, func_name, language, func, func_tokens, docstring in zip(
            df['repository_name'], df['func_code_url'], df['func_name'],
            df['language'], df['func_code_string'], df['func_code_tokens'],
            df['func_documentation_string']):
        # task = f"You are an expert Python developer and technical writer. Your task is to write concise and detailed Python docstring for the following function. Follow google-style format: \n{func}"
        print(f"Processing: {func_name}")
        match num:
            case "1":
                zero_doc = type_prompts.zero_shot_prompting(func, num_prompts)
            case "2":
                few_doc = type_prompts.few_shot_prompting(func, examples, num_prompts)
            case "3":
                chain_doc = type_prompts.chain_of_thought_prompting(func, num_prompts)
            case "4":
                structured_doc = type_prompts.structured_prompting(func, num_prompts)

            case "5":
                one_doc = type_prompts.one_shot_prompting(func, examples, num_prompts)

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
                    "prompt": prompt,
                    "generated_doc": zero_doc,

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
                    "prompt": prompt,
                    "generated_doc": few_doc,

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
                    "prompt": prompt,

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
                    "prompt": prompt,

                    "generated_doc": structured_doc,
                    "origin_doc": docstring
                })
            case "5":
                one_shot_records.append({
                    "filename": os.path.basename(file_path),
                    "function_name": func_name,
                    "model": model_used,
                    "prompt": prompt,

                    "input_code": func,
                    "generated_doc": one_doc,
                    "origin_doc":docstring
                })


    save_to_csv("output/generated_docs_.csv", prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n Saved dataset to output/generated_docs_.csv. Total runtime {elapsed_time} seconds")
