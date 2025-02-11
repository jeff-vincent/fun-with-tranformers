import os
import pandas as pd

output_filename = 'spanish-convos-simple.parquet'
filename_selector_segment = 'conversations' #example in which all parquet files include the characters 1000_

# column names from existing parquet files; add more as needed
column_a_name = 'speaker_a'
column_b_name = 'speaker_b'

def main():

    files = os.listdir('.')

    # Filter parquet files based on filename_selector_segment
    files = [f for f in files if filename_selector_segment in f]

    full_convo_dataset = []

    # Read each parquet file, extract speaker data, and combine into one DataFrame for final output.
    for f in files:
        df = pd.read_parquet(f, engine='fastparquet')
        speaker_a = None
        speaker_b = None

        for k, v in df.to_dict().items():
            if k == column_a_name:
                speaker_a = v
            elif k == column_b_name:
                speaker_b = v

        for k, v in speaker_a.items():
            entry = {}
            entry[column_a_name] = [v][0]
            entry[column_b_name] = speaker_b[k]
            full_convo_dataset.append(entry)

    df = pd.DataFrame.from_dict(full_convo_dataset)
    df.to_parquet(output_filename, engine='fastparquet')

if __name__ == '__main__':
    main()
