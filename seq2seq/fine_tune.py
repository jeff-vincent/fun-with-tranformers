# Lambda Labs imports: pip install datasets transformers tf_keras accelerate>=0.26.0

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Model and dataset configuration
model_name = 'google/flan-t5-small'
dataset_name = 'jeff-vincent/1k-spanish-convo-01'
output_path = './fine-tuned-flan-t5-small-spanish-convo-01'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split="train")

# Tokenize dataset with padding and truncation
def preprocess_function(examples):
    inputs = examples["student"]  # Input from "student" column
    targets = examples["tutor"]  # Output from "tutor" column
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

    # Tokenize target texts (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split into train and validation sets
split_data = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

# Set dataset format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
batch_size = 128
args = Seq2SeqTrainingArguments(
    output_path,
    evaluation_strategy = "epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model(output_path)

print(f"Training complete. Model saved at {output_path}")
