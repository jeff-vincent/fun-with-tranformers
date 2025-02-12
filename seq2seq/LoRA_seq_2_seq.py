import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = 'google/flan-t5-xl'
dataset_name = 'jeff-vincent/1k-spanish-tutor-corrections'
output_path = './LoRA-trained-spanish-tutor'

# Training arguments
batch_size = 8  
learning_rate = 3e-5 
num_train_epochs = 3
lora_ranks = 16  

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split="train")

# Tokenization function
def preprocess_function(examples):
    # Convert strings to lists for batching
    inputs = [str(e) for e in examples["student"]]
    targets = [str(t) for t in examples["tutor"]]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)
    
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=512, padding='max_length', truncation=True)

    # Replace pad tokens with -100 in labels to ignore them in loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Split into train and validation sets
split_data = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

# LoRA configuration
lora_config = LoraConfig(
    r=lora_ranks,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train and save
trainer.train()
trainer.save_model(output_path)

print(f"Training complete. Model saved at '{output_path}'")
