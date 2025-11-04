import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.title('Sentiment Demo')

mode = st.selectbox('Inference mode', ['local-baseline', 'local-transformer', 'sagemaker'])
text = st.text_area('Enter review text', height = 150)

if st.button ('Predict'):
  if mode == 'local-baseline':
    vec = joblib.load('models/tfidf_vectorizer.joblib')
    clf = joblib.load('models/logreg_baseline.joblib')
    X = vec.transform([text])
    pred = clf.predict_proba(X) [0,1]
    st.write('Positive probability:', float(pred))
  elif mode == 'local-transformer':
    tokenizer = Autotokenizer.from_pretrained ('models/distilbert-finetuned')
    model = AutoModelForSequenceClassification.from_pretrained('models/distilbert-finetuned')
    enc = tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 128)
    with torch.no_grad():
      logits = model(**enc).logits
    prob = torch.softmax(logits, dim = 1) [0,1].item()
    st.write ('Positive probability:', prob)
  else:
    st.write('SageMaker mode not configured in this template.')
    
