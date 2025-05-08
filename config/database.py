import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()

url = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password=os.getenv("PASSWORD"),
    host="localhost",
    port=5432,
    database="user_data"
)

engine = create_engine(url)
