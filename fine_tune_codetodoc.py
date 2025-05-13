import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainerCallback, EarlyStoppingCallback
from sentence_transformers import SentenceTransformer, util
from unsloth import FastLanguageModel
import optuna

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

base_path = "code-to-doc/data/codesearchnet/python/final/jsonl"
train_files = [os.path.join(base_path, "train", f) for f in os.listdir(os.path.join(base_path, "train")) if f.endswith(".jsonl")]
valid_files = [os.path.join(base_path, "valid", f) for f in os.listdir(os.path.join(base_path, "valid")) if f.endswith(".jsonl")]
test_files = [os.path.join(base_path, "test", f) for f in os.listdir(os.path.join(base_path, "test")) if f.endswith(".jsonl")]

def load_dataset(files):
    data = []
    for f in files:
        data.extend(load_jsonl(f))
    return [{"code": entry["code"], "docstring": entry["docstring"]} for entry in data if entry["code"] and entry["docstring"]]

train_data = load_dataset(train_files)
valid_data = load_dataset(valid_files)
test_data = load_dataset(test_files)

raw_datasets = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(valid_data),
    "test": Dataset.from_list(test_data)
})

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
    loftq_config=None
)

def tokenize_function(examples):
    return tokenizer(
        examples["code"],
        text_target=examples["docstring"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 300),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"])
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./unsloth-finetuned-code-doc",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True,
    predict_with_generate=True,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=4
)

trainer = Seq2SeqTrainer(
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=10
)

print("Best hyperparameters:", best_trial.hyperparameters)

final_args = training_args
final_args.learning_rate = best_trial.hyperparameters["learning_rate"]
final_args.per_device_train_batch_size = best_trial.hyperparameters["per_device_train_batch_size"]
final_args.weight_decay = best_trial.hyperparameters["weight_decay"]
final_args.num_train_epochs = best_trial.hyperparameters["num_train_epochs"]
final_args.warmup_steps = best_trial.hyperparameters["warmup_steps"]
final_args.lr_scheduler_type = best_trial.hyperparameters["lr_scheduler_type"]

final_trainer = Seq2SeqTrainer(
    model=model,
    args=final_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

final_trainer.train()

final_trainer.save_model("./final-unsloth-model")

print("\nRunning Semantic Similarity Evaluation...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

sample_test_data = raw_datasets["test"].select(range(20))
preds = final_trainer.predict(sample_test_data)
generated = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

references = sample_test_data["docstring"]
emb_gen = embedder.encode(generated, convert_to_tensor=True)
emb_ref = embedder.encode(references, convert_to_tensor=True)

cosine_scores = util.cos_sim(emb_gen, emb_ref).diagonal()

print("\nSemantic Similarity Scores (Cosine):")
print("Mean:", cosine_scores.mean().item())
print("Min:", cosine_scores.min().item())
print("Max:", cosine_scores.max().item())
print("Median:", cosine_scores.median().item())
