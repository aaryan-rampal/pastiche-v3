from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Float,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base
from datetime import datetime


class Artist(Base):
    """Artist table to store unique artists"""

    __tablename__ = "artists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    artworks = relationship("Artwork", back_populates="artist")


class Genre(Base):
    """Genre table to store art genres"""

    __tablename__ = "genres"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    artworks = relationship("Artwork", back_populates="genre")


class Artwork(Base):
    """Main artwork table combining data from both JSON and CSV"""

    __tablename__ = "artworks"

    id = Column(Integer, primary_key=True, index=True)

    # From artwork_s3_mapping.json
    artwork_key = Column(
        String(255), unique=True, nullable=False, index=True
    )  # The key from JSON
    s3_url = Column(Text, nullable=False)
    filename = Column(Text, nullable=False, index=True)
    description = Column(Text)

    # From classes_truncated.csv
    phash = Column(String(16), index=True)  # Perceptual hash
    width = Column(Integer)
    height = Column(Integer)
    genre_count = Column(Integer)
    subset = Column(String(10), index=True)  # train/test/val
    exists = Column(Boolean, default=True)

    # Foreign keys
    artist_id = Column(Integer, ForeignKey("artists.id"), nullable=False, index=True)
    genre_id = Column(Integer, ForeignKey("genres.id"), nullable=False, index=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    artist = relationship("Artist", back_populates="artworks")
    genre = relationship("Genre", back_populates="artworks")
    contours = relationship("Contour", back_populates="artwork")

    # Indexes for performance
    __table_args__ = (
        Index("idx_artist_genre", "artist_id", "genre_id"),
        Index("idx_dimensions", "width", "height"),
        Index("idx_subset_exists", "subset", "exists"),
    )


class Contour(Base):
    """Table to store contour data associated with artworks"""

    __tablename__ = "contours"

    id = Column(Integer, primary_key=True, index=True)
    artwork_id = Column(Integer, ForeignKey("artworks.id"), nullable=False, index=True)
    points = Column(Text, nullable=False)  # Store as JSON string
    image_shape = Column(
        String(50), nullable=False
    )  # e.g., "(height, width, channels)"
    created_at = Column(DateTime, default=func.now())

    # Relationships
    artwork = relationship("Artwork", back_populates="contours")

    # Indexes
    __table_args__ = (Index("idx_artwork_id", "artwork_id"),)
