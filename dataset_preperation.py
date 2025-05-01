from datasets import load_dataset
import json
import os
from itertools import islice


def build_instruction_response(example):
    code = example['func_code_string']
    doc = example['func_documentation_string']
    prompt = f"[Task] Let's understand this code together.\n\n[Code]\n{code}\n\n[Thought] Step-by-step, what is happening in this function?\n"
    return {"input": prompt, "output": doc.strip()}


def build_multi_turn_blocks(dataset, group_size=3):
    conversation_blocks = []
    iterator = iter(dataset)

    while True:
        batch = list(islice(iterator, group_size))
        if not batch:
            break

        messages = []
        for entry in batch:
            code = entry['func_code_string']
            doc = entry['func_documentation_string']
            messages.append({"role": "user", "content": f"Explain this function:\n{code}"})
            messages.append({"role": "assistant", "content": doc.strip()})

        conversation_blocks.append({"messages": messages})

    return conversation_blocks


def apply_chat_template(messages):
    input_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages[:-1]])
    output_text = messages[-1]['content']
    return {"input": input_text, "output": output_text}


def save_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + "\n")


def main():
    dataset = load_dataset("espejelomar/code_search_net_python_10000_examples", split="train")

    # Single-turn CoT style formatting
    simple_pairs = [build_instruction_response(x) for x in dataset]
    save_jsonl(simple_pairs, "output/fine_tune_cot.jsonl")

    # Multi-turn conversation formatting
    conversations = build_multi_turn_blocks(dataset, group_size=3)
    chat_formatted = [apply_chat_template(conv['messages']) for conv in conversations]
    save_jsonl(chat_formatted, "output/fine_tune_chat.jsonl")


if __name__ == '__main__':
    main()
