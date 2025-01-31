import openai
import pandas as pd

openai.api_key = ''


class SyntheticSpanishTutorData():
    def __init__(self, convo_theme):
        self.conversation = []
        self.convo_theme = convo_theme

    def create_entry(self, input):
         student_data = self.student(input)
         tutor_data = self.tutor(student_data)
         entry = {
              "student": student_data,
              "tutor": tutor_data
         }
         self.conversation.append(entry)
         return entry

    def student(self, response):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "assistant", "content": "You are learning Spanish. Your grammar is rarely correct, but you can express ideas reasonably clearly."}, 
                        {"role":"user", "content": f"With no greeting and with 150 words or fewer, respond to the following statement or question by giving a response or posing a question focusing on the converstaion theme '{self.convo_theme}': {response}"}],
        )
        return response.choices[0].message.content
    
    def tutor(self, response):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "assistant", "content": "You are a native Spanish speaker."}, 
                        {"role":"user", "content": f"With no greeting and with 150 words or fewer, respond to the following statement or question by giving a response or posing a question focusing on the converstaion theme '{self.convo_theme}': {response}"}],
        )
        return response.choices[0].message.content
    
    def write_to_parquet(self):
         df = pd.DataFrame.from_dict(self.conversation)
         df.to_parquet("1000_synthetic_spanish_tutor_data_peliculas_02.parquet")
         print("Synthetic Spanish Tutor Data saved to parquet file.")


def main():
     convo_theme = 'peliculas'
     convo = SyntheticSpanishTutorData(convo_theme=convo_theme)
     entry = f"Hoy, vamos a hablar sobre {convo_theme}"
     while len(convo.conversation) <= 1000:
        r = convo.create_entry(entry)
        print(r)
        entry = r['tutor']
            
     convo.write_to_parquet()


if __name__ == "__main__":
     main()