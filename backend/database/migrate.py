"""
Database migration script to create all tables
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database.connection import engine, Base
from database.models import Artist, Genre, Artwork, Contour, SearchQuery


def create_tables():
    """Create all database tables"""
    print("Creating database tables...")

    try:
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")

        # Print created tables
        inspector = engine.inspect(engine)
        tables = inspector.get_table_names()
        print(f"Created tables: {', '.join(tables)}")

    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

    return True


def drop_tables():
    """Drop all database tables (use with caution!)"""
    print("⚠️  WARNING: This will drop all tables and data!")
    confirm = input("Are you sure? Type 'yes' to continue: ")

    if confirm.lower() == "yes":
        try:
            Base.metadata.drop_all(bind=engine)
            print("All tables dropped successfully!")
            return True
        except Exception as e:
            print(f"Error dropping tables: {e}")
            return False
    else:
        print("Operation cancelled.")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        drop_tables()
    else:
        create_tables()
