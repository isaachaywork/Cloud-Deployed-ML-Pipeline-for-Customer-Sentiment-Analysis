import sagemaker
from sagemaker.huggingface import HuggingfaceModel
import boto3

def main():
  region = boto3.Session().region_name
  sagemaker_session = sagemaker.Session()
  role = sagemaker.get_execution_role()

  model_artifact = 's3://cloud-ml-pipeline-analysis/models/transformer/output/model.tar.gz'

  huggingface_model = HuggingfaceModel(
    model_data = midel_artifact,
    role = role,
    transformers_version = '4.26',
    pytorch_version = '1.13',
    py_version = 'py39'
  )

  predictor huggingface_model.deploy(
    initil_instnace_count = 1,
    instance_type = 'ml.m5.large',
    endpoint_name = 'sentiment-transformer-endpoint',
  )

  print ('deployment complete.')
  print (f'Endpoint name: {predictor.endpoint_name}')

if __name__ == '__main__':
  main()
