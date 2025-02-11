import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Model & Dataset Configuration
TEACHER_MODEL = "jeff-vincent/flan-t5-spanish-tutor"
STUDENT_MODEL = "google/flan-t5-small"
DATASET_NAME = "jeff-vincent/24k-spanish-convos-simple"
BATCH_SIZE = 4
EPOCHS = 1
LR = 5e-5
SAVE_PATH = "./distilled-flan-t5-small"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

# Load Models
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL).eval()
student_model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL).train()

# Load Dataset
dataset = load_dataset(DATASET_NAME, split="train")

# Move models to device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(DEVICE)
student_model.to(DEVICE)

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, device):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        inputs = str(example["speaker_a"])
        targets = str(example["speaker_b"])

        model_inputs = self.tokenizer(
            inputs, text_target=targets, max_length=512, truncation=True, return_tensors="pt", padding="max_length"
        )

        # Move to device
        for key in model_inputs:
            model_inputs[key] = model_inputs[key].squeeze(0).to(self.device)

        return model_inputs

# Initialize Dataset & DataLoader
train_dataset = TextDataset(dataset, tokenizer, DEVICE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer & Scheduler
optimizer = AdamW(student_model.parameters(), lr=LR)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * EPOCHS)

# Distillation Training Loop
for epoch in range(EPOCHS):
    student_model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Get Teacher Outputs (No Gradient Needed)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            teacher_logits = teacher_outputs.logits

        # Get Student Outputs
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        student_logits = student_outputs.logits

        # Compute Loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)), 
            labels.view(-1), 
            ignore_index=tokenizer.pad_token_id
        )
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        )

        loss = 0.5 * ce_loss + 0.5 * kl_loss  # Weighted loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss=loss.item())

# Save the Fine-Tuned Student Model
student_model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"🎉 Distilled model saved at {SAVE_PATH}")
