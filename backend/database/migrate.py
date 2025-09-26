"""
Database migration script to create all tables
"""

import sys
from loguru import logger
from sqlalchemy import inspect
from connection import engine, Base


def create_tables():
    """Create all database tables"""
    logger.debug("Creating database tables...")

    try:
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created successfully!")

        # Print created tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Created tables: {', '.join(tables)}")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
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
