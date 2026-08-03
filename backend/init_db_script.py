import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.init_db import init_db

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized!")
