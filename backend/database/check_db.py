"""
Script to check database contents and display statistics
"""

import sys
from loguru import logger
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database.connection import get_db_session
from database.models import Artist, Genre, Artwork, Contour, SearchQuery
from sqlalchemy import func


def check_database_contents():
    """Check and display database contents"""
    logger.info("🔍 Checking database contents...\n")

    with get_db_session() as db:
        # Basic counts
        logger.info("📊 Table Counts:")
        logger.info(f"   Artists: {db.query(Artist).count()}")
        logger.info(f"   Genres: {db.query(Genre).count()}")
        logger.info(f"   Artworks: {db.query(Artwork).count()}")
        logger.info(f"   Contours: {db.query(Contour).count()}")
        logger.info(f"   Search Queries: {db.query(SearchQuery).count()}")

        logger.info("\n" + "=" * 50)

        # Top 10 artists by artwork count
        logger.info("\n👨‍🎨 Top 10 Artists by Artwork Count:")
        top_artists = (
            db.query(Artist.name, func.count(Artwork.id).label("artwork_count"))
            .join(Artwork)
            .group_by(Artist.name)
            .order_by(func.count(Artwork.id).desc())
            .limit(10)
            .all()
        )

        for i, (artist, count) in enumerate(top_artists, 1):
            logger.info(f"   {i:2d}. {artist}: {count} artworks")

        # Top genres
        logger.info("\n🎨 Genres by Artwork Count:")
        genres = (
            db.query(Genre.name, func.count(Artwork.id).label("artwork_count"))
            .join(Artwork)
            .group_by(Genre.name)
            .order_by(func.count(Artwork.id).desc())
            .all()
        )

        for i, (genre, count) in enumerate(genres, 1):
            logger.info(f"   {i:2d}. {genre}: {count} artworks")

        # Dataset splits
        logger.info("\n📚 Dataset Splits:")
        splits = (
            db.query(Artwork.subset, func.count(Artwork.id).label("count"))
            .group_by(Artwork.subset)
            .order_by(func.count(Artwork.id).desc())
            .all()
        )

        for subset, count in splits:
            logger.info(f"   {subset}: {count} artworks")

        # Sample artworks
        logger.info("\n🖼️  Sample Artworks:")
        sample_artworks = db.query(Artwork).join(Artist).join(Genre).limit(5).all()

        for artwork in sample_artworks:
            logger.info(f"   • {artwork.artist.name} - '{artwork.description}'")
            logger.info(
                f"     Genre: {artwork.genre.name}, Dimensions: {artwork.width}x{artwork.height}"
            )
            logger.info(f"     S3 URL: {artwork.s3_url[:60]}...")

        # Image dimensions stats
        logger.info("\n📐 Image Dimensions Statistics:")
        dimension_stats = db.query(
            func.min(Artwork.width).label("min_width"),
            func.max(Artwork.width).label("max_width"),
            func.avg(Artwork.width).label("avg_width"),
            func.min(Artwork.height).label("min_height"),
            func.max(Artwork.height).label("max_height"),
            func.avg(Artwork.height).label("avg_height"),
        ).first()

        logger.info(
            f"   Width:  Min={dimension_stats.min_width}, Max={dimension_stats.max_width}, Avg={dimension_stats.avg_width:.1f}"
        )
        logger.info(
            f"   Height: Min={dimension_stats.min_height}, Max={dimension_stats.max_height}, Avg={dimension_stats.avg_height:.1f}"
        )

        # Check for any issues
        logger.info("\n🔍 Data Quality Checks:")
        missing_s3 = db.query(Artwork).filter(Artwork.s3_url.is_(None)).count()
        missing_dimensions = (
            db.query(Artwork)
            .filter((Artwork.width.is_(None)) | (Artwork.height.is_(None)))
            .count()
        )
        not_exists = db.query(Artwork).filter(Artwork.exists == False).count()

        logger.info(f"   Missing S3 URLs: {missing_s3}")
        logger.info(f"   Missing dimensions: {missing_dimensions}")
        logger.info(f"   Files marked as not existing: {not_exists}")


def search_artworks_by_artist(artist_name: str):
    """Search for artworks by a specific artist"""
    logger.info(f"\n🔍 Searching for artworks by '{artist_name}':")

    with get_db_session() as db:
        artworks = (
            db.query(Artwork)
            .join(Artist)
            .filter(Artist.name.ilike(f"%{artist_name}%"))
            .limit(10)
            .all()
        )

        if not artworks:
            logger.info(f"   No artworks found for artist '{artist_name}'")
            return

        for artwork in artworks:
            logger.info(f"   • {artwork.description}")
            logger.info(
                f"     Dimensions: {artwork.width}x{artwork.height}, Genre: {artwork.genre.name}"
            )


def search_artworks_by_genre(genre_name: str):
    """Search for artworks by genre"""
    logger.info(f"\n🔍 Searching for artworks in genre '{genre_name}':")

    with get_db_session() as db:
        artworks = (
            db.query(Artwork)
            .join(Genre)
            .filter(Genre.name.ilike(f"%{genre_name}%"))
            .limit(10)
            .all()
        )

        if not artworks:
            logger.info(f"   No artworks found for genre '{genre_name}'")
            return

        for artwork in artworks:
            logger.info(f"   • {artwork.artist.name} - {artwork.description}")
            logger.info(f"     Dimensions: {artwork.width}x{artwork.height}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            command = sys.argv[1]
            if command == "artist" and len(sys.argv) > 2:
                search_artworks_by_artist(sys.argv[2])
            elif command == "genre" and len(sys.argv) > 2:
                search_artworks_by_genre(sys.argv[2])
            else:
                logger.info("Usage:")
                logger.info(
                    "  python check_db.py                    # Show all statistics"
                )
                logger.info(
                    "  python check_db.py artist 'Van Gogh'  # Search by artist"
                )
                logger.info("  python check_db.py genre 'Impression' # Search by genre")
        else:
            check_database_contents()
    except Exception as e:
        logger.info(f"❌ Error checking database: {e}")
        import traceback

        traceback.print_exc()
