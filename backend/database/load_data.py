"""
Data loading script to populate the database from JSON and CSV files
"""

import json
import csv
import sys
import os
from typing import Dict, Any
from loguru import logger
import re
from sqlalchemy import func

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database.connection import get_db_session
from database.models import Artist, Genre, Artwork


def clean_genre_string(genre_str: str) -> str:
    """Clean genre string and extract the first genre"""
    # Remove brackets and quotes, then take the first genre
    cleaned = genre_str.strip("[]'\"")
    # Split by comma and take the first genre
    first_genre = cleaned.split(",")[0].strip("'\"")
    return first_genre


def load_artwork_data():
    """Load artwork data from JSON and CSV files into the database"""

    # File paths
    json_file = "../../s3_setup/artwork_s3_mapping.json"
    csv_file = "../../data/classes_truncated.csv"

    logger.debug("Loading artwork data into database...")

    try:
        # Load JSON data
        logger.debug("Reading JSON file...")
        with open(json_file, "r") as f:
            json_data = json.load(f)

        # Load CSV data
        logger.debug("Reading CSV file...")
        csv_data = {}
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_data[row["filename"]] = row

        logger.debug(
            f"Loaded {len(json_data)} JSON records and {len(csv_data)} CSV records"
        )

        # Start database session
        with get_db_session() as db:
            # Track artists and genres to avoid duplicates
            artists_cache = {}
            genres_cache = {}

            # Process each artwork
            processed = 0
            skipped = 0

            for artwork_key, json_artwork in json_data.items():
                try:
                    filename = json_artwork["filename"]

                    # Find corresponding CSV data
                    csv_artwork = csv_data.get(filename)
                    if not csv_artwork:
                        logger.warning(f"No CSV data for {filename}")
                        skipped += 1
                        continue

                    # Get or create artist
                    artist_name = json_artwork["artist"]
                    if artist_name not in artists_cache:
                        artist = (
                            db.query(Artist).filter(Artist.name == artist_name).first()
                        )
                        if not artist:
                            artist = Artist(name=artist_name)
                            db.add(artist)
                            db.flush()  # Get the ID
                        artists_cache[artist_name] = artist.id

                    artist_id = artists_cache[artist_name]

                    # Get or create genre
                    genre_str = clean_genre_string(json_artwork["genre"])
                    if genre_str not in genres_cache:
                        genre = db.query(Genre).filter(Genre.name == genre_str).first()
                        if not genre:
                            genre = Genre(name=genre_str)
                            db.add(genre)
                            db.flush()  # Get the ID
                        genres_cache[genre_str] = genre.id

                    genre_id = genres_cache[genre_str]

                    # Create artwork record
                    artwork = Artwork(
                        artwork_key=artwork_key,
                        s3_url=json_artwork["s3_url"],
                        filename=filename,
                        description=json_artwork["description"],
                        phash=csv_artwork["phash"],
                        width=int(csv_artwork["width"]),
                        height=int(csv_artwork["height"]),
                        genre_count=int(csv_artwork["genre_count"]),
                        artist_id=artist_id,
                        genre_id=genre_id,
                    )

                    db.add(artwork)
                    processed += 1

                    # Commit in batches
                    if processed % 100 == 0:
                        db.commit()
                        logger.debug(f"Processed {processed} artworks...")

                except Exception as e:
                    logger.error(f"Error processing {artwork_key}: {e}")
                    skipped += 1
                    continue

            # Final commit
            db.commit()

            logger.info(f"Successfully processed {processed} artworks")
            logger.warning(f"Skipped {skipped} artworks")
            logger.info(f"Created {len(artists_cache)} unique artists")
            logger.info(f"Created {len(genres_cache)} unique genres")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

    return True


def get_database_stats():
    """Get statistics about the loaded data"""
    with get_db_session() as db:
        artist_count = db.query(Artist).count()
        genre_count = db.query(Genre).count()
        artwork_count = db.query(Artwork).count()

        logger.info("\n📊 Database Statistics:")
        logger.info(f"   Artists: {artist_count}")
        logger.info(f"   Genres: {genre_count}")
        logger.info(f"   Artworks: {artwork_count}")

        # Top genres
        top_genres = (
            db.query(Genre.name, func.count(Artwork.id).label("count"))
            .join(Artwork)
            .group_by(Genre.name)
            .order_by(func.count(Artwork.id).desc())
            .limit(5)
            .all()
        )

        logger.info("\n🎨 Top 5 Genres:")
        for genre, count in top_genres:
            logger.info(f"   {genre}: {count} artworks")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        get_database_stats()
    else:
        if load_artwork_data():
            get_database_stats()
