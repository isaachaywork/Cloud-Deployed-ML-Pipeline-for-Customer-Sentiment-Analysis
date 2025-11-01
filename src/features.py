import joblib
import pandas as pd
from sklearn,feature_extraction.text import TfidfVectorizer

def fit_tfidf(train_path = 'data/train.parquet'):
  df = pd.read_parquet(train_path)
  vec = TfidfVectorizer(max_features=20000, ngram_range(1,2))
  X = vec.fit_transform(df['text_clean'])
  joblib.dump(vec, 'models/tfidf_vectorizer.joblib')
  print('Saved TF - IDF vectorizer')

if __name__ == '__main__':
  fit_tfidf()
