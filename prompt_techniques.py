import requests
import json

class PromptTechniques:
    def __init__(self, model):
        self.model = model

    def generate_with_ollama(self, prompt):

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
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
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        return result.strip() or "[No response from model]"

    def zero_shot_prompting(self, task_description):
        prompt = task_description
        return self.generate_with_ollama(prompt)

    def one_shot_prompting(self, task_description, example):
        """
        One-shot prompting: include a single example before the task.
        """
        prompt = (
            f"Here is one example:\n"
            f"Function:\n{example['input']}\n"
            f"Docstring:\n{example['output']}\n\n"
            f"Now, write a docstring for the following function:\n{task_description}\n"
            f"Output:"
        )
        return self.generate_with_ollama(prompt)

    def few_shot_prompting(self, task_description, examples):
        examples_text = "examples"
        for example in examples:
            examples_text += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        prompt = f"""
        Here are some examples:
        {examples_text}
        Now, do the following task:
        Task: {task_description}
        Output:
        """
        return self.generate_with_ollama(prompt)

    def chain_of_thought_prompting(self, problem, code_snippet):
        prompt = f"""
        Problem: {problem}
        Let's think through this step by step to find the answer for the following function {code_snippet}:
        """
        return self.generate_with_ollama(prompt)

    def structured_prompting(self, code_snippet):
        prompt = f"""You are a senior Python developer and technical writer.

        Your task is to generate a complete, professional docstring for the following Python function.
        Use Google-style formatting. Include:
    
        - A one-line summary of what the function does
        - Args section: each argument, its type, and purpose
        - Returns section: return value, type, and purpose
        - Raises section if applicable
        - Include an example if the function is non-obvious
    
        Function to document:
        ```python
        {code_snippet}"""
        return self.generate_with_ollama(prompt)