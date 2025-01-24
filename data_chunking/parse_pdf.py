import PyPDF2
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000):
    """Chunk the extracted text into smaller sections."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def save_chunks_to_csv(chunks, output_file="chunked_text.csv"):
    """Save the chunks into a CSV file."""
    # Create a DataFrame from the chunks
    df = pd.DataFrame(chunks, columns=["Text"])
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Text has been chunked and saved into {output_file}.")

def main():
    pdf_path = 'us-history-chapter-1.pdf'  # Replace with your PDF path
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    save_chunks_to_csv(chunks)
    
if __name__ == "__main__":
    main()
