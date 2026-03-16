import boto3
import json
from cinerag import config
from pathlib import Path
from typing import Literal

s3_client = boto3.client("s3")


def upload_processed_jsonl(
    data: list[dict],
    s3_file_name: str = f"{config.DEFAULT_FILE_NAME}.jsonl",
    bucket: str = config.DEFAULT_S3_BUCKET,
) -> None:

    jsonl_data = "\n".join(json.dumps(record) for record in data)
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{config.S3_PROCESSED_DATA_FOLDER}{s3_file_name}",
            Body=jsonl_data,
            ContentType="application/json",
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")


def upload_raw_file(
    file_path: Path,
    s3_file_name: str = f"{config.DEFAULT_FILE_NAME}.jsonl",
    bucket: str = config.DEFAULT_S3_BUCKET,
) -> None:

    try:
        s3_client.upload_file(
            Filename=file_path,
            Bucket=bucket,
            Key=f"{config.S3_RAW_DATA_FOLDER}{s3_file_name}",
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")


def file_exists(
    s3_file_name: str,
    type: Literal["raw", "processed", "embeddings"],
    bucket: str = config.DEFAULT_S3_BUCKET,
) -> bool:

    key = (
        config.S3_RAW_DATA_FOLDER
        if type.strip() == "raw"
        else (
            config.S3_PROCESSED_DATA_FOLDER
            if type.strip() == "processed"
            else config.S3_EMBEDDINGS_FOLDER
        )
    )
    key = key + s3_file_name
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False


def get_processed_data_stream(
    s3_file_name: str = f"{config.DEFAULT_FILE_NAME}.jsonl",
    bucket: str = config.DEFAULT_S3_BUCKET,
):
    try:
        response = s3_client.get_object(
            Bucket=bucket, Key=f"{config.S3_PROCESSED_DATA_FOLDER}{s3_file_name}"
        )
        return response
    except Exception as e:
        print(f"Error fetching file from S3: {e}")
        return None
