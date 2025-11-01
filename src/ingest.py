import argparse
import os
import pandas as pd
import boto3

def local_sample():
  df = pd.read_csv('data/sample_reviews.csv')
  df.to_parquet('data/clean_reviews.parquet', index = False)
  print('Wrote data/clean_reviews.parquet')

def upload_s3(bucket, key, filename):
  s3 = boto3.client('S3')
  s3.upload_file(filename, bucket, key)
  print(f'Uploaded {filename} to s3://{bucket}/{key}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--upload-s3', action='store_true')
  parser.add_argument('--bucket')
  parser.add_argument('--key', default='data/clean_reviews.parquet')
  args = parser.parse_args()
  local_sample()
  if args.upload_s3:
    assert args.bucket
    upload_s3(args.bucket, args.key, 'data/clean_reviews.parquet')
    

