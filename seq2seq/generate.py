from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "jeff-vincent/flan-t5-spanish-tutor"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize input and move it to the same device as the model
input_text = "me gusta mango, pero prefiero manzanas. Y tu, que tipas de frutas te gustan?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate output
output_ids = model.generate(
    input_ids,
    max_length=520,
    min_length=500,
    do_sample=True
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)