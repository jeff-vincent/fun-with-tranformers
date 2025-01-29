import openai
import pandas as pd

openai.api_key = ''

class GenerateQuestionsFromChunkedText:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.chunked_text = None
        self.qa_context = []

    def set_chunked_text(self, text):
        df = pd.read_parquet(self.parquet_path)
        self.chunked_text = df['text'].tolist()
        
    def generate_questions(self):
        for i, chunk in enumerate(self.chunked_text):
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "developer", "content": "You are a helpful assistant."}, 
                          {"role":"user", "content": f"Generate one question and answer set drawn from the content in this text that resembles as closely as possible a real search query. Return it in the following format: Question: Answer:\n\nContext: {chunk}"}],
            )
            qa_set = response.choices[0]
            content = qa_set.message.content
            try:
                question = content.split("\n")[0].strip().replace("Question: ", "")
                answer = content.split("\n")[2].strip().replace("Answer: ", "")
                self.qa_context.append({"question": question, "answer": answer, "context": chunk})
            except IndexError:
                print(f"Error parsing question and answer from chunk {i+1}. Skipping...")

    def write_to_parquet(self):
        df = pd.DataFrame.from_dict(self.qa_context)
        df.to_parquet(f"{self.parquet_path.split('.')[0]}_questions.parquet")
        print("Questions generated and saved to parquet file.")


def main():
    parquet_path = "/Users/jeffvincent/fun-with-tranformers/qa_from_chunks/astromony-textbook.parquet"
    generator = GenerateQuestionsFromChunkedText(parquet_path)
    generator.set_chunked_text(parquet_path)
    generator.generate_questions()
    generator.write_to_parquet()


if __name__ == "__main__":
    main()
