import spacy
import pyarrow as pa
import pyarrow.parquet as pq
import math

# Function to split text into chunks of approximately 300 words
def split_into_chunks(text, chunk_size=300):
    nlp = spacy.blank("en")  # Use a blank spaCy model for tokenization
    doc = nlp(text)
    
    words = [token.text for token in doc if not token.is_space]
    total_words = len(words)
    
    # Calculate the number of chunks needed
    num_chunks = math.ceil(total_words / chunk_size)
    
    chunks = [" ".join(words[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_chunks)]
    return chunks

# Read input text file
input_file = "text.txt"  # Replace with your input text file name
output_file = "open-stax-us-history-chapter-1.parquet"  # Output Parquet file name

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Chunk the text
chunks = split_into_chunks(text, chunk_size=300)

# Create a Parquet table
table = pa.Table.from_pydict({"text": chunks})

# Write to a Parquet file
pq.write_table(table, output_file)

print(f"Chunks saved to {output_file}")
