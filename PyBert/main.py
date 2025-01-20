from transformers import AutoTokenizer, AutoModelForCausalLM
import tensorflow as tf

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Sample input with a masked token
input_text = "Where would you like to go for dinner?"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Forward pass through the model
outputs = model(**inputs)

# Decode the output
decoded_tokens = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
print(decoded_tokens)

from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
dataset = load_dataset("flytech/python-codes-25k")

# Define the tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],          # Tokenize only the 'text' column
        padding="max_length",      # Pad sequences to the max length
        truncation=True,           # Truncate sequences longer than max_length
        max_length=736             # Fixed length for all sequences
    )

# Tokenize the dataset
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names  # Keep only the tokenized data
)

# Output tokenized dataset structure for verification
print(tokenized_datasets)

from transformers import Trainer, TrainingArguments

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="no",     # evaluation strategy to use during training
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    logging_steps=200,               # log every 200 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=tokenized_datasets["train"],  # training dataset
    # eval_dataset=tokenized_datasets["test"],   # evaluation dataset
)

# Start fine-tuning
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_distilgpt2")
tokenizer.save_pretrained("./fine_tuned_distilgpt2")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
import random

# # Step 1: Load a Pretrained Model and Tokenizer
# model_name = "distilbert-base-uncased"  # Replaceable with similar-sized models
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Load a Compatible Dataset
# Using a dataset with Python code examples (flytech dataset)
dataset = load_dataset("flytech/python-codes-25k", split="train")

# Step 3: Add Random Labels (Binary Classification Example)
# For practice, add random labels to the dataset
def add_labels(example):
    example["labels"] = random.randint(0, 1)  # Assign random 0 or 1
    return example

dataset = dataset.map(add_labels)

# Step 4: Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],  # Use the text field from the dataset
        padding="max_length",
        truncation=True,
        max_length=128  # Adjust as needed for shorter or longer inputs
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Data Collation
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # Step 6: Define the Model
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 7: Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,  # Keep only the last 2 checkpoints
    report_to="none",  # Avoid logging to external services
)

# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    eval_dataset=tokenized_dataset,  # For evaluation, if needed
)

# Step 9: Train the Model
trainer.train()

# Step 10: Save the Model and Tokenizer
model.save_pretrained("./fine_tuned_distilbert")
tokenizer.save_pretrained("./fine_tuned_distilbert")

print("Fine-tuning complete! Model saved to './fine_tuned_distilbert'.")

