import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, DataCollatorForSeq2Seq)
from sentence_transformers import SentenceTransformer

# Model and dataset configuration
model_name = '/content/drive/MyDrive/colab_output/flan-t5-spanish-tutor-02'
dataset_name = 'jeff-vincent/1k-spanish-convo-14'
output_path = '/content/drive/MyDrive/colab_output/flan-t5-spanish-tutor-02'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load topic embedding model for topic consistency loss
topic_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
batch_size = 4
args = Seq2SeqTrainingArguments(
    output_path,
    evaluation_strategy="epoch",
    learning_rate=1e-6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Custom loss function to enforce topic consistency
def topic_consistency_loss(generated_texts, target_texts, topic_embedding_model):
    """
    Penalizes responses that are semantically distant from the expected output.
    Uses cosine similarity on sentence embeddings.
    """
    gen_emb = topic_embedding_model.encode(generated_texts, convert_to_tensor=True)
    tgt_emb = topic_embedding_model.encode(target_texts, convert_to_tensor=True)
    
    loss = 1 - F.cosine_similarity(gen_emb, tgt_emb).mean()
    return loss

# Custom trainer class with topic consistency loss
class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard Seq2Seq loss (cross-entropy)
        ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=tokenizer.pad_token_id)

        # Generate model predictions
        generated_tokens = model.generate(inputs["input_ids"], max_length=50)
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        target_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute topic consistency loss
        tc_loss = topic_consistency_loss(generated_texts, target_texts, topic_embedding_model)

        # Combine losses
        total_loss = ce_loss + 0.5 * tc_loss  # Adjust weight of topic consistency loss as needed

        return (total_loss, outputs) if return_outputs else total_loss

trainer = CustomTrainer(
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
