import openai
import pandas as pd

openai.api_key = ''
from topics_2 import topics_1, topics_2, topics_3, topics_4, topics_5, topics_6, topics_7, topics_8, topics_9, topics_10

class SyntheticSpanishTutorDataWithContext():
    def __init__(self, convo_theme, context):
        self.conversation = []
        self.convo_theme = convo_theme
        self.context = context

    def create_entry(self, input):
         student_data = self.student(input)
         tutor_data = self.tutor(student_data)
         entry = {
              "student": f"{student_data}\n[Context]\n{self.context}",
              "tutor": tutor_data
         }
         self.conversation.append(entry)
         return entry

    def student(self, response):
        prompt = f"""
                With a focus on {self.convo_theme} and with no greeting, in 150 words or fewer, respond to the following statement or question by giving a response or posing a question focusing on the following context:

                [Context]
                {self.context}

                [Question or Comment]
                {response}
                """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "assistant", "content": "You are learning Spanish. Your grammar is rarely correct, but you can express ideas reasonably clearly."}, 
                        {"role":"user", "content": prompt}],
        )
        return response.choices[0].message.content
    
    def tutor(self, response):
        prompt = f"""
                In Spanish, and with a focus on {self.convo_theme} and with no greeting, in 150 words or fewer, respond to the following statement or question by giving a response and posing a question focusing on the following context:

                [Context]
                {self.context}

                [Question or Comment]
                {response}
                """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "assistant", "content": "You are a native Spanish speaker."}, 
                        {"role":"user", "content": prompt}],
        )
        return response.choices[0].message.content
    
    def write_to_parquet(self):
         df = pd.DataFrame.from_dict(self.conversation)
         df.to_parquet(f"spanish_{self.convo_theme.replace(' ', '_')}_topic_convo_with_context.parquet")
         print("Synthetic Spanish Tutor Data saved to parquet file.")

def get_context(convo_theme):
    prompt = f"write a 150 word summary of {convo_theme} and include key Spanish language vocabulary words for discussion on the topic."
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "assistant", "content": "You are a helpful assistant."}, 
                    {"role":"user", "content": prompt}],
    )
    return response.choices[0].message.content

def main():
     for item in topics_10:
        convo_theme = item
        context = get_context(convo_theme)
        convo = SyntheticSpanishTutorDataWithContext(convo_theme=convo_theme, context=context)
        entry = f"Hoy, vamos a hablar sobre {convo_theme}"
        while len(convo.conversation) <= 10:
            r = convo.create_entry(entry)
            print(r)
            entry = r['tutor']
                
        convo.write_to_parquet()


if __name__ == "__main__":
     main()
