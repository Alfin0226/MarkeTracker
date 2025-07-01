import os 
from dotenv import load_dotenv
import psycopg
from companies import COMPANY_DATA

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")

with psycopg.connect(POSTGRES_URL) as conn:
    with conn.cursor() as cur:
        for company in COMPANY_DATA:
            cur.execute(
                """
                INSERT INTO companies (name, symbol)
                VALUES (%s, %s)
                ON CONFLICT (symbol) DO NOTHING
                """,
                (company["name"], company["symbol"])
            )
        conn.commit()
print("Companies inserted!")