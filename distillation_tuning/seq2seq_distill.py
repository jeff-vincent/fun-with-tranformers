import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AdamW,
)
from datasets import load_dataset
from tqdm import tqdm

# Model and dataset configuration
teacher_model_name = './fine-tuned-flan-t5-large-spanish-convo-01'
student_model_name = 'google/flan-t5-small'
dataset_name = 'jeff-vincent/1k-spanish-convo-01'
output_path = './distil-tuned-flan-t5-small-spanish-convo-01'

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name).to("cuda")
student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name).to("cuda")

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Tokenization function with proper padding and tensor output
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["student"],
        truncation=True,
        padding="max_length",  # Ensures all sequences have the same length
        max_length=128,
        return_tensors="pt",  # Force PyTorch tensors
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tutor"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",  # Convert to tensors
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Convert to PyTorch tensors explicitly
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset into train and validation sets
split_data = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

# Data collator ensures proper batching
data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)

# DataLoader with collate function to handle batch conversion
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

# Optimizer
optimizer = AdamW(student_model.parameters(), lr=2e-3)

# Training loop with mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
num_epochs = 50
alpha = 0.5
temperature = 2.0

for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
    
        # Generate decoder input ids (shifted right labels)
        decoder_input_ids = student_model.prepare_decoder_input_ids_from_labels(labels)
    
        # Get teacher logits
        with torch.no_grad():
            teacher_logits = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  # Explicitly passing decoder inputs
            ).logits
    
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            student_logits = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  # Explicitly passing decoder inputs
            ).logits

            loss_kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id)
            loss = alpha * loss_kl + (1 - alpha) * loss_ce

        total_loss += loss.item()

        # Backpropagation with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        progress_bar.set_postfix({"Loss": loss.item()})

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")

# Save the student model and tokenizer
student_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"Training complete. Model saved at {output_path}")
