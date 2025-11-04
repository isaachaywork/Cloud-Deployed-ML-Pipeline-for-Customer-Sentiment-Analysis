from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd

def prepare_dataset (parquet_path):
  df = pd.read_parquet(parquet_path)[['text_clean','label']]
  return Dataset.from_pandas(df)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = np.argmax(pred.predictions, axis=1)
  from sklearn.metrics import f1_score, accuracy_score
  return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds)}

def main():
  tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
  model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)
  train_ds = prepare_dataset('data/train.parquet')
  val_ds = prepare_dataset('data/val.parquet')
  def tokenize(batch): return tokenizer(batch['text_clean'], truncation = True, padding = 'max_length', max_length = 128)
  train_ds = train_ds.map (tokenize, batched = True)
  val_ds = val_ds.map(tokenize, batched = True)
  train_ds.set_format(type='torch', columns =['input_ids','attention_mask','label'])
  val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
  args = TrainingArguments(
    output_dir='models/distilbert-finetuned',
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    num_train_epochs = 2,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    logging_steps = 50
  )
  trainer = Trainer(model=model, args=args, train_dataset = train_ds, eval_dataset=val_ds, compute_metrics = compute_metrics)
  trainer.train()
  trainer.save_model('models/distilbert-finetuned')
  print('Saved transformer to models/distilbert-finetuned')

if __name__ == '__main__':
  main()
