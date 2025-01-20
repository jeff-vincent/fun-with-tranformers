from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, DataCollatorForSeq2Seq
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

# Wrap models for multi-GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    student_model = torch.nn.DataParallel(student_model)
    teacher_model = torch.nn.DataParallel(teacher_model)

teacher_model.to(device)
student_model.to(device)

# Load a language modeling dataset
dataset = load_dataset("izumi-lab/open-text-books", split="train")

# Estimate maximum token length and filter long examples
max_allowed_tokens = 512  # Set to 512 based on model and dataset average

# Filter out examples with more than max_allowed_tokens
filtered_dataset = dataset.filter(
    lambda example: len(tokenizer(example["text"], truncation=False)["input_ids"]) <= max_allowed_tokens
)

# Tokenize dataset with dynamic padding
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=max_allowed_tokens)

tokenized_dataset = filtered_dataset.map(preprocess_function, batched=True)

# DataCollator to handle dynamic padding during training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)

# DataLoader
train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    # Soft target loss (KL Divergence)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss_kl = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    # Hard target loss (Cross-Entropy)
    loss_ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),  # Use reshape instead of view
        labels.reshape(-1),  # Use reshape instead of view
        ignore_index=tokenizer.pad_token_id
    )

    # Weighted combination of losses
    return alpha * loss_kl + (1 - alpha) * loss_ce

# Optimizer
optimizer = AdamW(student_model.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
alpha = 0.5
temperature = 2.0

for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        # Prepare inputs
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        # Get teacher logits
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        # Get student logits
        student_logits = student_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        # Compute loss
        loss = distillation_loss(student_logits, teacher_logits, labels, alpha, temperature)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"Loss": loss.item()})

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")

# Save model and tokenizer
student_model.module.save_pretrained("/models/distil_tuned_flan-t5-large") if isinstance(student_model, torch.nn.DataParallel) else student_model.save_pretrained("./distil_tuned_flan-t5-large")
tokenizer.save_pretrained("./distil_tuned_flan-t5-large")
