
'''
https://console.aws.amazon.com/
https://console.aws.amazon.com/sagemaker
'''

# test reading text file in s3
import boto3
s3 = boto3.client("s3")
bucket = "sagemaker-studio-147795258718-ywchj6yljj8"
key = "test.txt"
# Read the file
response = s3.get_object(Bucket=bucket, Key=key)
content = response['Body'].read().decode("utf-8")
print(content)
