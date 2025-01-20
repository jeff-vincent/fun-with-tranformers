from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

torch.cuda.empty_cache()

student_model_name = "google/flan-t5-large"
teacher_model_name = "google/flan-t5-xl"

student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name)
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name)
teacher_model.gradient_checkpointing = False

tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# Load a language modeling dataset (e.g., WikiText-103, OpenWebText)
dataset = load_dataset("izumi-lab/open-text-books", split="train")

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

# Tokenize and limit dataset to first 10 lines
tokenized_dataset = dataset.map(preprocess_function)
# train_dataset = tokenized_datasets["text"] # .select(range(10))  # Select first 10 lines

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    # Soft target loss (KL Divergence)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss_kl = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    # Hard target loss (Cross-Entropy)
    loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id)

    # Weighted combination of losses
    return alpha * loss_kl + (1 - alpha) * loss_ce

# Create data loader
train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# Optimizer
optimizer = AdamW(student_model.parameters(), lr=1e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

num_epochs = 1
alpha = 0.5  # Weight for soft loss
temperature = 2.0  # Temperature for soft targets

for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0

    # Add tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        # If input_ids is a list of tensors, stack them
        if isinstance(batch["input_ids"], list):
            input_ids = torch.stack(batch["input_ids"]).to(device)  # Stack the list of tensors into one tensor
        else:
            input_ids = batch["input_ids"].to(device)  # If it's already a tensor, just move it to the device

        # Prepare decoder input ids
        decoder_input_ids = input_ids[:, :-1]  # Exclude the last token
        labels = input_ids[:, 1:]  # Exclude the first token

        # Get teacher logits
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        # Get student logits
        student_logits = student_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits, labels, alpha, temperature)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update the progress bar with the current loss
        progress_bar.set_postfix({"Loss": loss.item()})

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")

# Save the student model and tokenizer to verify file writing
student_model.save_pretrained("./distil_tuned_flan-t5-large")
tokenizer.save_pretrained("./distil_tuned_flan-t5-large")
