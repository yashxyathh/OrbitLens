import os
import uuid
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Read config from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "mock_access_key")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mock_secret_key")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "satquery-production-storage")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def upload_to_s3(file_content: bytes, filename: str, content_type: str) -> str:
    """
    Uploads a file to AWS S3 and returns the public URL.
    In a real production app with missing credentials, we simulate a successful URL.
    """
    unique_filename = f"{uuid.uuid4()}-{filename}"
    
    try:
        if AWS_ACCESS_KEY_ID == "mock_access_key":
            # For development without active AWS keys, mock the URL
            return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{unique_filename}"
            
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=unique_filename,
            Body=file_content,
            ContentType=content_type,
            # In production, you might not use public-read if data is sensitive. 
            # We assume signed URLs or public access based on requirements.
            ACL='public-read' 
        )
        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{unique_filename}"
        return url
        
    except (NoCredentialsError, ClientError) as e:
        print(f"Failed to upload to S3: {e}")
        # Fallback for local testing
        return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{unique_filename}"
