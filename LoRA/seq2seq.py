import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = 'google/flan-t5-xl'
dataset_name = 'izumi-lab/open-text-books'
output_path = f'/LoRA-tuned-{model_name.split("/")[1]}-{dataset_name.split("/")[1]}'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split="train")

# Estimate maximum token length and filter long examples
max_allowed_tokens = 512  # Set to 512 based on model and dataset average

# Filter out examples with more than max_allowed_tokens
filtered_dataset = dataset.filter(
    lambda example: len(tokenizer(example["text"], truncation=False)["input_ids"]) <= max_allowed_tokens
)

# Tokenize dataset with padding and truncation
def preprocess_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_allowed_tokens)
    tokens["labels"] = tokens["input_ids"].copy()  # Add labels for CLM
    return tokens

tokenized_dataset = filtered_dataset.map(preprocess_function, batched=True)

# Split into train and validation sets
split_data = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

# Take only 3 or 4 items for quick testing
train_dataset = train_dataset.select(range(3))  # Take the first 3 items
eval_dataset = eval_dataset.select(range(4))

# Set dataset format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=".",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",  # Disable reporting to external tools like WandB
    load_best_model_at_end=True,
    save_strategy="steps",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the LoRA fine-tuned model
trainer.save_model("./saved_model")

print(f"Training complete. Model saved at {output_path}")
