import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# PostgreSQL
POSTGRES_URI = os.getenv("POSTGRES_URI")
