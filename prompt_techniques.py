import requests
import json

class PromptTechniques:
    def __init__(self, model, max_retries=3, initial_timeout=30):
        self.model = model
        self.max_retries = max_retries
        self.initial_timeout = initial_timeout

        # Zero Shot Prompts
        self.zero_shot_prompts = [
            f"As a senior Python engineer, write a professional Google-style docstring for this function: {{code}}",
            f"Please document the following Python function using Google-style format, providing detailed parameter and return info: {{code}}",
            f"Generate a structured Python docstring in Google format for the following function. Explain inputs, outputs, and purpose: {{code}}",
            f"You are tasked with documenting the next function. Use Google-style formatting and ensure clarity and completeness: {{code}}",
            f"Write a clear and concise Google-style docstring for this function as if preparing code for production: {{code}}",
            f"You are a Python developer writing internal documentation. Add a complete Google-style docstring to the function below: {{code}}",
            f"As part of code review, produce a Google-style docstring for this function explaining its behavior: {{code}}",
            f"Document the Python function below using the Google style. Include a summary, arguments, return value, and exceptions if any: {{code}}",
            f"Write documentation for the following function using the Google-style docstring standard. Be informative and concise: {{code}}",
            f"Imagine you're writing docs for a public API. Add a proper Google-style docstring to the function: {{code}}"
            f"Generate a one-line docstring describing what this Python function does: {{code}}",
            f"Write a Google-style docstring for this function including Args and Returns sections: {{code}}",
            f"Create a NumPy-style docstring for this function with Parameters, Returns, and Examples sections: {{code}}",
            f"Document this function including: 1) Purpose 2) Parameters 3) Return value 4) Usage example: {{code}}",
            f"Generate documentation for this function that explicitly lists edge cases: {{code}}",
            f"Write Python type-annotated docstring including parameter and return types: {{code}}",
            f"Create documentation for this class method including its relationship to class state: {{code}}",
            f"Document this async function explaining its awaitable behavior: {{code}}",
            f"Write docs for this generator function explaining its yielding behavior: {{code}}",
            f"Explain what this decorator does and show example usage: {{code}}",
            f"Write production-quality documentation for this mission-critical function: {{code}}",
            f"Create beginner-friendly documentation with simple explanations: {{code}}",
            f"Document this function including performance characteristics: {{code}}",
            f"Write documentation for this data science function explaining its math operations: {{code}}",
            f"Document this web handler including expected request/response formats: {{code}}",
            f"Create docs for this ML function explaining algorithm and hyperparameters: {{code}}",
            f"Write a concise docstring for this function in exactly 3 sentences: {{code}}",
            f"Document this function by contrasting it with [standard_library_function]: {{code}}",
            f"Create security-aware documentation highlighting potential vulnerabilities: {{code}}"
        ]

        # Chain Of Thought Prompts
        self.chain_of_thought_prompts = [
            f"Analyze this function step-by-step then write a Google-style docstring: 1) Purpose 2) Parameters 3) Returns 4) Examples 5) Docstring: {{code}}",
            f"As a senior engineer, document this function through: 1) Understanding logic 2) Identifying inputs/outputs 3) Noting edge cases 4) Writing docs: {{code}}",
            f"Think through: 1) What does this function do? 2) What does it need? 3) What does it return? 4) Now write professional docs: {{code}}",
            f"Documentation process: 1) Analyze function signature 2) Understand implementation 3) Note special cases 4) Write Google-style docs: {{code}}",
            f"Step-by-step: 1) Parse function name 2) Examine parameters 3) Study return value 4) Write complete docstring: {{code}}",
            f"Reason through: 1) Function purpose 2) Parameter roles 3) Return value meaning 4) Produce documentation: {{code}}",
            f"As a Python expert, analyze then document: 1) Overall behavior 2) Input requirements 3) Output details 4) Final docstring: {{code}}",
            f"Systematic documentation: 1) Understand code 2) Identify key aspects 3) Consider usage 4) Write professional docs: {{code}}",
            f"Professional doc process: 1) Study implementation 2) Note important details 3) Structure information 4) Write Google docs: {{code}}",
            f"Methodical approach: 1) What problem solves? 2) How implemented? 3) What returns? 4) Write formal docs: {{code}}",
            f"Code review style: 1) Verify function 2) Check parameters 3) Validate returns 4) Write complete docs: {{code}}",
            f"Engineering docs: 1) Requirements 2) Design 3) Verification 4) Write specs: {{code}}",
            f"Teach this function: 1) Concepts needed 2) How it works 3) How to use 4) Write docs: {{code}}",
            f"Production docs: 1) Business need 2) Technical solution 3) Usage patterns 4) Write formally: {{code}}",
            f"API documentation: 1) Interface contract 2) Error conditions 3) Examples 4) Write docs: {{code}}",
            f"Complete analysis: 1) Functional role 2) Data flow 3) Edge cases 4) Write professional docs: {{code}}",
            f"Deep dive: 1) Code walkthrough 2) Key operations 3) Failure modes 4) Write documentation: {{code}}",
            f"Quality docs: 1) Correctness 2) Safety 3) Performance 4) Write complete docs: {{code}}",
            f"Maintainer's view: 1) Evolution path 2) Dependencies 3) Technical debt 4) Write docs: {{code}}",
            f"Expert documentation: 1) Core algorithm 2) Optimization 3) Tradeoffs 4) Write formal docs: {{code}}"
        ]
        
    # Zero Shot Prompting
    def zero_shot_prompting(self, code, num_prompts):
        selected = self.zero_shot_prompts[:num_prompts]
        return [{
            "prompt": p.format(code=code),
            "output": self.generate_with_ollama(p.format(code=code))
        } for p in selected]

    def one_shot_prompting(self, code, examples, num_prompts):
        """
        One-shot prompting: Run `num_prompts` separate one-shot prompts using distinct examples.
        """
        selected_examples = examples[:num_prompts]
        results = []

        #create multiple prompts with different examples
        for example in selected_examples:
            prompt = (
                f"Here is one example:\n"
                f"Function:\n{example['input']}\n"
                f"Docstring:\n{example['output']}\n\n"
                f"Now, write a docstring for the following function:\n{code}\n"
                f"Output:"
            )
            output = self.generate_with_ollama(prompt)
            results.append({
                "prompt": prompt,
                "model_output": output
            })

        return results

    def few_shot_prompting(self, code, examples, num_prompts):
        """
        Few-shot prompting: include multiple examples before the task.
        """
        definition = ("Google-style docstrings are a structured format for documenting Python code, "
                      "they follow a specific format including a summary line, detailed explanations, "
                      "and dedicated sections for parameters, return values, and exceptions.")
        selected_examples = examples[:num_prompts]
        result = []
        examples_text = ""
        for ex in selected_examples:
            examples_text += f"Function:\n{ex['input']}\nDocstring:\n{ex['output']}\n\n"
            prompt = (
                f"{definition}\n\n"
                f"Examples of Google-style docstrings:\n\n"
                f"{examples_text}"
                f"Now, write a docstring for the following function:\n{code}\nOutput:"
            )
            output = self.generate_with_ollama(prompt)
            result.append({
                "prompt": prompt,
                "model_output": output
            })
        return result

    # Chain Of Thought Prompting
    def chain_of_thought_prompting(self, code_snippet, num_prompts):
        selected_prompts = self.chain_of_thought_prompts[:num_prompts]

        results = []
        for template in selected_prompts:
            prompt = template.format(code=code_snippet)
            output = self.generate_with_ollama(prompt)
            results.append({
                "reasoning_framework": template.split('\n')[0][:50] + "...",
                "prompt": prompt,
                "model_output": output
            })

        return results

    def structured_prompting(self, code, num_prompts):
        prompts = [
            f"""You are a senior Python developer and technical writer.

        Your task is to generate a complete, professional docstring for the following Python function.
        Use Google-style formatting. Include:

        - A one-line summary of what the function does
        - Args section: each argument, its type, and purpose
        - Returns section: return value, type, and purpose
        - Raises section if applicable
        - Include an example if the function is non-obvious

        Function to document:
        ```python
        {{code}}""",

            f"""You are a Python expert responsible for writing clear and maintainable documentation.

        Please write a Google-style docstring for the following function. The docstring should include:
        - A brief summary of the function's purpose
        - Arguments with their types and descriptions
        - Return value details
        - Any exceptions raised
        - An example usage if helpful

        Function to document:
        ```python
        {{code}}""",

            f"""As a lead software engineer, produce a professional Google-style docstring for the function below.

        Your documentation must include:
        - A summary line
        - Arguments and their types
        - Return value and its purpose
        - Exception details if applicable
        - Usage example when appropriate

        Function to document:
        ```python
        {{code}}""",

            f"""You are creating developer-facing documentation. Generate a complete docstring using Google format for the function below.

        The docstring should contain:
        - A concise one-line summary
        - Args section with types and meanings
        - Returns section
        - Raises section if needed
        - An illustrative example if applicable

        Function to document:
        ```python
        {{code}}""",

            f"""Write a comprehensive Python docstring in Google style for the following function.

        Include:
        - Summary
        - Arguments and their explanations
        - Return information
        - Exceptions raised (if any)
        - An example call if the function behavior isn't obvious

        Function to document:
        ```python
        {{code}}""",

            f"""You are documenting this function for inclusion in a company’s internal codebase.

        Use the Google Python docstring format and include:
        - What the function does
        - Its parameters (name, type, purpose)
        - Its return value (type and description)
        - Exceptions it might raise
        - A short example if non-trivial

        Function to document:
        ```python
        {{code}}""",

            f"""As a documentation engineer, create a detailed Google-style docstring for the function provided.

        Make sure the docstring includes:
        - A descriptive summary line
        - Arguments with types and purposes
        - A Returns section
        - A Raises section (if needed)
        - An example for clarity

        Function to document:
        ```python
        {{code}}""",

            f"""Create a full Python docstring for the function below using the Google docstring format.

        You must include:
        - One-line function summary
        - Args: argument name, type, and description
        - Returns: type and meaning
        - Raises: when applicable
        - A usage example if needed

        Function to document:
        ```python
        {{code}}""",

            f"""Write a docstring in Google style for the function shown.

        Document:
        - What the function does
        - Each parameter and its role
        - The return value and its type
        - Any exceptions
        - Example usage if helpful

        Function to document:
        ```python
        {{code}}""",

            f"""You're preparing this function for open-source release.

        Generate a thorough Google-style docstring that includes:
        - A one-liner describing the function
        - Parameter types and descriptions
        - Return value details
        - Exceptions (if applicable)
        - A usage example to demonstrate intent

        Function to document:
        ```python
        {{code}}""",

            f"""You are reviewing this code for documentation standards.

        Write a high-quality docstring in Google format including:
        - Summary of behavior
        - Arguments with type and meaning
        - Return value
        - Raised exceptions (if any)
        - Example usage if non-obvious

        Function to document:
        ```python
        {{code}}"""
        ]
        selected = prompts[:num_prompts]
        result = []
        for prompt in selected:
            output = self.generate_with_ollama(prompt)
            result.append({
                "prompt": prompt,
                "output": output,
            })
        return result