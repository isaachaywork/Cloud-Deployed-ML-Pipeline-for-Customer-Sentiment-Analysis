import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(s: str) -> str:
  s = s.lower()
  s = re.sub(r"http\S+",'',s)
  s = re.sub(r'[^a-z0-9\s]','',s)
  s = re.sub(r'\s+','',s).strip()
  return s

def run(inp='data/clean_reviews.parquet'):
  df = pd.read_parquet(inp)
  df['text_clean'] = df['review_text'].fillna('').map(clean_txt)
  df['label'] = (df['rating'] >= 4).astype(int)
  train, temp = train_test_split(df, test_size = 0.3, random_state = 42, stratify=df['label'])
  val, test = train_test_split(temp, test_size=0.5, random_state = 42, stratify=temp['label'])
  train.to_parquet('data/train.parquet', index = False)
  val.to_parquet('data/val.parquet', index = False)
  test.to_parquet('data/test,parquet', index = False)
  print('Wrote train/val/test to data/')

if __name__ == '__main__':
  run()
