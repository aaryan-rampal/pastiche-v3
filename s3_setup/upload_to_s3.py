import boto3
import dotenv
import os
from typing import Dict
from pathlib import Path
import json
from tqdm import tqdm
import hashlib
import pandas as pd


class S3ArtworkUploader:
    """Handle uploading artwork images to S3 from CSV file"""

    def __init__(
        self,
        bucket_name: str,
        region: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_default_region: str,
    ):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists")
        except Exception:
            print(f"Creating bucket {self.bucket_name}")
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )

    def generate_artwork_id(self, filename: str, local_path: str) -> str:
        """Generate a unique ID for artwork based on filename and file hash"""
        # Use file hash for uniqueness
        with open(local_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]

        # Create clean filename without extension
        clean_name = Path(filename).stem
        return f"{clean_name}_{file_hash}"

    def upload_image(self, local_path: str, s3_key: str, artwork_id: str) -> str:
        """Upload single image to S3 with specific key and return URL"""
        try:
            # Upload to S3 with the exact folder structure
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )

            # Return S3 URL
            s3_url = (
                f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            )
            return s3_url

        except Exception as e:
            print(f"Error uploading {local_path}: {e}")
            return ""

    def upload_from_csv(self, csv_path: str, base_dir: str) -> Dict[str, dict]:
        """Upload all images from CSV file maintaining folder structure"""

        # Read CSV
        print(f"Reading CSV from {csv_path}")
        df = pd.read_csv(csv_path)

        # Filter only existing files
        df_exists = df[df["exists"] == True].copy()
        print(f"Found {len(df_exists)} existing images to upload")

        mapping = {}
        failed_uploads = []

        # Upload each image
        for idx, row in tqdm(
            df_exists.iterrows(), total=len(df_exists), desc="Uploading images"
        ):
            filename = row["filename"]

            # Construct full local path
            local_path = os.path.join(base_dir, filename)

            # Verify file exists locally
            if not os.path.exists(local_path):
                print(f"File not found locally: {local_path}")
                failed_uploads.append(filename)
                continue

            # Generate artwork ID
            artwork_id = self.generate_artwork_id(filename, local_path)

            # Use filename as S3 key to preserve folder structure
            s3_key = f"artworks/{filename}"

            # Upload image
            s3_url = self.upload_image(local_path, s3_key, artwork_id)

            if s3_url:
                mapping[artwork_id] = {
                    "s3_url": s3_url,
                    "filename": filename,
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "description": row["description"],
                }
            else:
                failed_uploads.append(filename)

        # Report results
        print(f"Successfully uploaded: {len(mapping)}")
        print(f"Failed uploads: {len(failed_uploads)}")

        if failed_uploads:
            print("Failed files:")
            for failed in failed_uploads[:10]:  # Show first 10
                print(f"  - {failed}")
            if len(failed_uploads) > 10:
                print(f"  ... and {len(failed_uploads) - 10} more")

        return mapping

    def save_mapping_to_file(self, mapping: Dict[str, Dict], output_file: str):
        """Save the artwork_id -> metadata mapping to JSON file"""
        with open(output_file, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved mapping to {output_file}")


def main():
    """Main upload script"""
    # AWS Credentials from environment variables
    dotenv.load_dotenv()  # Load environment variables from .env file
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

    # Configuration
    BUCKET_NAME = "wikiart-pastiche"
    REGION = "us-east-2"
    CSV_PATH = "../data/classes_truncated.csv"  # Path to CSV
    BASE_DIR = "/Volumes/Extreme SSD/wikiart/"  # Base directory for images
    OUTPUT_MAPPING_FILE = "artwork_s3_mapping.json"

    # Initialize uploader
    uploader = S3ArtworkUploader(
        BUCKET_NAME,
        REGION,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_DEFAULT_REGION,
    )

    # Create bucket if needed
    uploader.create_bucket_if_not_exists()

    # Upload all images from CSV
    print("Starting CSV-based upload...")
    mapping = uploader.upload_from_csv(CSV_PATH, BASE_DIR)

    # Save mapping
    uploader.save_mapping_to_file(mapping, OUTPUT_MAPPING_FILE)

    print(f"Upload complete! {len(mapping)} images uploaded.")
    print(f"Mapping saved to {OUTPUT_MAPPING_FILE}")


if __name__ == "__main__":
    main()
