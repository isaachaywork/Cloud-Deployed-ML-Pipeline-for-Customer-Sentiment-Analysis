import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from features import fit_tfidf

def train():
  fit_tfidf()
  vec = joblib.load('models/tfidf_vectorizer.joblib')
  df = pd.read_parquet('data/train.parquet')
  X = vec.transform(df['text_clean'])
  y = df['label']
  clf = LogisticRegression(max_iter = 1000)
  clf.fit(X,y)
  joblib.dump (clf, 'models/logreg_baseline.joblib')
  print ('Saved models/logreg_baseline.joblib')

if __name__ == '__main__':
  train()
