aws s3 mb s3://cloud-ml-pipeline-analysis --region us-east-2
aws s3api put-bucket-versioning --bucket cloud-ml-pipeline-analysis --version-configuration Status=Enabled
