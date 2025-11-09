import sagemaker
from sagemaker.huggingface import Huggingface
import boto3
import os

def main():
  region = boto3.Session().region_name
  sagemaker_session = sagemaker.Session()
  role = sagemaker.get_execution_role()

  bucket = 'cloud-ml-pipeline-analysis'
  prefix = 'models/transformer'
  s3_output = f's3://{bucket}/{prefix}/output'

  script_path = 'scr/train_transformer.py'
  s3_code_path = sagemaker_session.upload_data(path=script_path, bucket = bucket, key_prefix = f'{prefix}/code')

  hyperparameters = {
  'epochs': 3,
  'train_batch_size' : 16,
  'model_name' : 'distilbert-base-uncased',
  'learning_rate' : 2e-5,
  }

  huggingface_estimator = HuggingFace(
    entry_point = 'train_transformer/py',
    source_dir = 'src',
    instance_type = 'ml.m5.xlarge',
    instance_count = 1,
    role = role,
    transformers_version = '4.26',
    pytorch_version = '1.13',
    py_version = 'py39',
    hyperparameters = hyperparameters,
    output_path = s3_output,
  )

  huggingface_estimator.fit({
    'train' : f's3://{bucket}/data/train.csv',
    'test' : f's3:/{bucket}/data/test.csv'
  })

  print('Training complete.')
  print('f'Model artifacts saved to: {s3_output}')

if __name__ == '__main__':
  main()
