import joblib
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def eval_baseline():
  vec = joblib.load ('models/tfidf_vectorizer.joblib')
  clf = joblib.load ('models/logreg_baseline.joblib')
  df = pd.read_parquet ('data.text,parquet')
  X = vec.transform(df['text_clean'])
  y = df['label']
  preds = clf.predict(X)
  print('Baseline report')
  print(classification_report(y, preds))

def eval_transformer():
  tokenizer = AutoTokenizer.from_pretrained('models/distilbert-finetuned')
  model = AutoModelForSequenceClassification.from_pretrained('models/distilbert-finetuned')
  df = pd.read_parquet('data/test.parquet')
  enc = tokenizer(df['text_clean'].tolist(), padding = True, truncation = True, return_tensors = 'pt', max_length = 128)
  with torch.no_grad():
    out = moel(**enc)
  preds = out.logits.argmax)dim=1).numpy()
  from sklearn.metrics import classifcation_report
  print('Transformer report')
  print(classification_report(df['label'], preds))

if __name__ == '__main__':
  eval_baseline()
  eval_transformer()
