# Pastiche Database Setup

This directory contains the database setup and models for the Pastiche application.

## Prerequisites

1. **PostgreSQL**: Install PostgreSQL on your system
2. **Python packages**: Install required packages:
   ```bash
   pip install sqlalchemy psycopg2-binary
   ```

## Database Setup

### 1. Create PostgreSQL Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE pastiche_db;

# Create user (optional)
CREATE USER pastiche_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE pastiche_db TO pastiche_user;
```

### 2. Environment Variables

Create a `.env` file in the backend directory or set environment variable:

```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/pastiche_db"
```

### 3. Create Tables

```bash
cd backend/database
python migrate.py
```

### 4. Load Data

```bash
python load_data.py
```

### 5. View Statistics

```bash
python load_data.py stats
```

## Database Schema

### Tables

1. **artists** - Stores unique artists

   - id, name, created_at, updated_at

2. **genres** - Stores art genres/movements

   - id, name, created_at

3. **artworks** - Main table combining JSON and CSV data

   - id, artwork_key, s3_url, filename, description
   - phash, width, height, genre_count, subset, exists
   - artist_id (FK), genre_id (FK)
   - created_at, updated_at

4. **contours** - Stores contour data for similarity search

   - id, artwork_id (FK), points_json, hu_moments
   - feature_vector, contour_count, extraction_method
   - created_at, updated_at

5. **search_queries** - Analytics for search queries
   - id, query_type, execution_time_ms, results_count
   - input_points_json, results_json, created_at

## Usage Examples

### Query artworks by artist:

```python
from database.connection import get_db_session
from database.models import Artist, Artwork

with get_db_session() as db:
    artworks = db.query(Artwork)\
                .join(Artist)\
                .filter(Artist.name == "Vincent Van Gogh")\
                .all()
```

### Query artworks by genre:

```python
with get_db_session() as db:
    impressionist_works = db.query(Artwork)\
                           .join(Genre)\
                           .filter(Genre.name == "Impressionism")\
                           .all()
```

## Files

- `connection.py` - Database connection and session management
- `models.py` - SQLAlchemy models/tables
- `migrate.py` - Table creation/migration script
- `load_data.py` - Data loading from JSON and CSV files
