import boto3, pathlib

model_dir = pathlib.Path("/mnt/models/")   # where your .bin / safetensors live
bucket = "models"
prefix = "roses/"

s3 = boto3.client(
        "s3",
        endpoint_url="<S3 URL>",
        aws_access_key_id='<S3 CLIENT>',
        aws_secret_access_key='<S3 SECRET>')

for f in model_dir.iterdir():
    if f.is_file():
        s3.upload_file(str(f), bucket, prefix + f.name)
