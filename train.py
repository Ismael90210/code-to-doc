from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from src.model import load_model_and_tokenizer


def build_sft_trainer(model, tokenizer, train_dataset, max_seq_length, output_dir):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to=[],
        ),
    )


def main():
    # Load the fine-tuning dataset
    dataset = load_dataset(
        "csv",
        data_files="output/generated_docs_.csv",
        split="train"
    )

    # 2) Model selection
    print("Available models:")
    print("  llama3.2:1b, qwen2.5-coder:0.5b, deepseek-r1:1.5b")
    model_name = input("Enter the model name to use: ")

    # 3) Load tokenizer and model via your loader
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Build the trainer
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=512,
        output_dir="output/checkpoints"
    )

    # Start fine-tuning
    trainer.train()


if __name__ == "__main__":
    main()
