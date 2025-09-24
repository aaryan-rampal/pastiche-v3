# S3 Setup for Pastiche

This directory contains scripts to upload your artwork collection to AWS S3 and set up DynamoDB for metadata storage.

## Prerequisites

1. **AWS Account**: You need an AWS account with appropriate permissions
2. **AWS CLI**: Install and configure AWS CLI
   ```bash
   brew install awscli
   aws configure
   ```
3. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Files Overview

- `upload_to_s3.py` - Main upload script that reads from CSV and uploads to S3
- `setup_dynamodb.py` - Sets up DynamoDB table and populates with metadata
- `setup_aws.sh` - Bash script to create AWS resources (bucket + table)
- `requirements.txt` - Python dependencies

## Usage

### 1. Set up AWS Infrastructure (Optional)

```bash
chmod +x setup_aws.sh
./setup_aws.sh
```

### 2. Upload Images to S3

```bash
python upload_to_s3.py
```

This script will:

- Read from `../data/classes_truncated.csv`
- Upload images from `/Volumes/Extreme SSD/wikiart/`
- Preserve folder structure in S3 as `artworks/{genre}/{filename}`
- Generate unique artwork IDs
- Save mapping to `artwork_s3_mapping.json`

### 3. Set up DynamoDB

```bash
python setup_dynamodb.py
```

This will:

- Create DynamoDB table `pastiche-artworks`
- Load artwork metadata from the mapping file
- Set up artwork_id → s3_url + metadata relationships

## Configuration

Edit these variables in `upload_to_s3.py`:

- `BUCKET_NAME`: Your S3 bucket name
- `REGION`: AWS region (default: us-east-1)
- `CSV_PATH`: Path to your CSV file
- `BASE_DIR`: Base directory where images are stored

## Output

After running, you'll have:

- All images uploaded to S3 with preserved folder structure
- `artwork_s3_mapping.json` with artwork_id → metadata mapping
- DynamoDB table with searchable artwork metadata

## Structure in S3

```
s3://your-bucket/
├── artworks/
│   ├── Baroque/
│   │   ├── rembrandt_painting.jpg
│   │   └── caravaggio_work.jpg
│   ├── Impressionism/
│   │   └── monet_waterlilies.jpg
│   └── ...
```

## Artwork ID Format

Each artwork gets a unique ID: `{clean_filename}_{hash8}`

- Example: `starry-night-1889_a1b2c3d4`

This ensures no duplicates even if filenames are similar across genres.
