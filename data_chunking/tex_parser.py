import argparse
import sys
import spacy
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text

class GutenbergParser:
    def __init__(self, filename):
        self.filename = filename
        self.text = None
        self.chunks = []
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.max_length = 15000000
        self.chunk_size = 1000

    def parse_tex(self):
        with open(self.filename, 'r') as file:
            self.text = LatexNodes2Text().latex_to_text(file.read())     

    def chunk(self):
        doc = self.nlp(self.text)
        chunk = ''
        for sent in doc.sents:
            clean_sent = sent.text.replace('\n', '')
            if len(clean_sent) < 25:
              continue
            if len(clean_sent) > 200:
              continue
            if len(chunk.replace('\n', '')) + len(clean_sent) > self.chunk_size:
                self.chunks.append(chunk.replace('\n', ''))
                chunk = ''
            chunk = chunk + f' {clean_sent}'
            chunk = chunk.replace('\n', '')

    def print_some_chunks(self):
      count = 0
      for chunk in self.chunks:
        count += 1
        if count <= 15:
          print('**************************')
          print(chunk)
          print('**************************')
        
    def write_to_parquet(self):
        self.chunks = self.chunks[:-100]
        df = pd.DataFrame({'text': self.chunks})
        df.to_parquet(f'{self.filename.split(".")[0]}.parquet')

def parse_args(args):
    parser = argparse.ArgumentParser(description='Gutenberg Text Parser')
    parser.add_argument('filename', help='Path to the Gutenberg text file')
    parser.add_argument('format', help='Input file format')
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    parser = GutenbergParser(args.filename)
    if args.format == 'tex':
        parser.parse_tex()
    elif args.format == 'plaintext':
        with open(args.filename, 'r') as file:
            parser.text = file.read()
    parser.chunk()
    parser.print_some_chunks()
    parser.write_to_parquet()

if __name__ == "__main__":
    main()
