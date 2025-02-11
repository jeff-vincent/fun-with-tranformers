import openai
import pandas as pd
import json

openai.api_key = ''

from topics_2 import topics_10 as topics
def generate(topic):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "assistant", "content": "You are a helpful assistant"}, 
                        {"role":"user", "content": f"Write a spanish conversation consisting of 50 exchanges between two speakers. Format the conversation as json. Each object in the list should represent one conversational exchange. Each object in the list must have the key `speaker_a` for one speaker and `speaker_b` for the second speaker. The conversation should focus on the topic of {topic}. Return only json; no additional text. Do not apply syntax highlighting. Return a simple string."}],
        )
        print(response.choices[0].message.content)
        convo = json.loads(response.choices[0].message.content)
        return convo

def main():
    for topic in topics:
          r = generate(topic)
          df = pd.DataFrame.from_dict(r)
          df.to_parquet(f"{topic.replace(' ', '_')}_spanish_conversations-1.parquet")


if __name__ == "__main__":
    main()