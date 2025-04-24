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

    def chain_of_thought_prompting(self, problem):
        prompt = f"""
        Problem: {problem}
        Let's think through this step by step to find the answer:
        """
        return self.generate_with_ollama(prompt)

    def structured_prompting(self, code_snippet):
        prompt = f"""
        You are an expert Python developer and technical writer. Your task is to write a professional, concise, and Google-style docstring for the following function:

        ```python
        {code_snippet}
        ```

        GENERATED DOCUMENTATION:
        """
        return self.generate_with_ollama(prompt)
