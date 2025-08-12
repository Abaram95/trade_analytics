from sqlalchemy import text
import pandas as pd
from engine import engine

def get_data(ticker: str):
    query = text("""
        SELECT day_date, close
        FROM stocks
        WHERE ticker = :ticker
          AND day_date >= CURRENT_DATE - INTERVAL '5 years'
        ORDER BY day_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"ticker": ticker})

def get_company_info(ticker: str):
    query = text("SELECT * FROM companies WHERE ticker = :ticker LIMIT 1")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    return df.iloc[0] if not df.empty else None

def get_tickers():
    query = text("SELECT ticker, long_name FROM companies ORDER BY ticker")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return [] if df.empty else df

def get_filters():
    query = text("SELECT DISTINCT country, sector FROM companies ORDER BY country, sector")
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def get_tickers_filtered_optional(country=None, sector=None):
    base_query = "SELECT ticker, long_name FROM companies"
    filters = []
    params = {}

    if country:
        filters.append("country = :country")
        params["country"] = country
    if sector:
        filters.append("sector = :sector")
        params["sector"] = sector
    if filters:
        base_query += " WHERE " + " AND ".join(filters)
    base_query += " ORDER BY ticker"

    query = text(base_query)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)
