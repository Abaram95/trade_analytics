from sqlalchemy import create_engine
from dotenv import load_dotenv
import os


load_dotenv()

DATABASE_URL = os.getenv("NEONBASE_URL")

if DATABASE_URL is None:
    raise Exception("DATABASE_URL não encontrada no .env")

engine = create_engine(DATABASE_URL, echo=False)


