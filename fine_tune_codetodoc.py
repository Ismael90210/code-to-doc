import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainerCallback, EarlyStoppingCallback
from sentence_transformers import SentenceTransformer, util
import optuna

# Step 1: Load data from JSONL files
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Step 2: Create a combined dataset (train/valid/test)
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

# Step 3: Convert to Hugging Face Dataset
raw_datasets = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(valid_data),
    "test": Dataset.from_list(test_data)
})

# Step 4: Tokenization
tokenizer = AutoTokenizer.from_pretrained("kdf/python-docstring-generation")

def tokenize_function(examples):
    return tokenizer(
        examples["code"],
        text_target=examples["docstring"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Map the tokenization
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Step 5: Load Model
model = AutoModelForSeq2SeqLM.from_pretrained("kdf/python-docstring-generation")

# Step 6: Define objective for hyperparameter tuning
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained("kdf/python-docstring-generation")

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"])
    }

# Step 7: Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-code-doc",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True,
    predict_with_generate=True,
    logging_dir="./logs",
    fp16=True
)

# Step 8: Trainer Setup
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Step 9: Hyperparameter search
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=10
)

print("Best hyperparameters:", best_trial.hyperparameters)

# Step 10: Final training with best hyperparameters
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

# Save final model
final_trainer.save_model("./final-finetuned-code-doc")

# Step 11: Semantic Similarity Evaluation
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
