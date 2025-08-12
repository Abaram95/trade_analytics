# fetch_stock.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import Table, MetaData, text
from sqlalchemy.dialects.postgresql import insert
from core.database.engine import engine
from math import log

def get_latest_date(ticker, engine):
    q = text("SELECT MAX(day_date) FROM stocks WHERE ticker = :ticker")
    with engine.connect() as conn:
        r = conn.execute(q, {"ticker": ticker}).scalar()
    if r is None:
        return datetime(2015, 1, 1).date()
    if isinstance(r, datetime):
        return r.date()
    return r

def get_last_close(ticker, engine):
    q = text("""
        SELECT close
        FROM stocks
        WHERE ticker = :ticker
        ORDER BY day_date DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        v = conn.execute(q, {"ticker": ticker}).scalar()
    return float(v) if v is not None else None

def fetch_and_store(tickers):
    if isinstance(tickers, str):
        tickers = [tickers]

    metadata = MetaData()
    table = Table("stocks", metadata, autoload_with=engine)

    records = []
    stats = {}  # sempre retornado

    for ticker in tickers:
        try:
            start_date = get_latest_date(ticker, engine) + timedelta(days=1)
            end_date = datetime.today().date()

            if start_date > end_date:
                print(f"⏩ {ticker} já está atualizado.")
                stats[ticker] = {
                    "status": "ok", "inserted": 0, "skipped": 1,
                    "start": start_date, "end": end_date
                }
                continue

            print(f"🔄 Baixando {ticker} de {start_date} até {end_date}...")

            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                progress=False,
                threads=True,
                group_by="ticker"
            )

            if df.empty:
                print(f"⚠️ Nenhum dado para {ticker}")
                stats[ticker] = {
                    "status": "ok", "inserted": 0, "skipped": 1,
                    "start": start_date, "end": end_date
                }
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df = df[ticker]

            df = df.reset_index()[["Date", "Close", "Volume"]]
            df.columns = ["day_date", "close", "volume"]
            df["ticker"] = ticker
            df["day_date"] = pd.to_datetime(df["day_date"]).dt.date

            # --- log_return usando último close do DB para o 1º registro do lote ---
            prev_close = get_last_close(ticker, engine)
            lr = []
            prev = prev_close
            for c in df["close"].astype(float):
                if prev is None:
                    lr.append(None)  # 1º histórico desse ticker
                else:
                    lr.append(log(c) - log(prev))
                prev = c
            df["log_return"] = lr
            # ----------------------------------------------------------------------

            records.extend(df.to_dict(orient="records"))
            stats[ticker] = {
                "status": "ok",
                "inserted": len(df),
                "skipped": 0,
                "start": start_date,
                "end": end_date
            }
            print(f"✅ {ticker} preparado para inserção. ({len(df)} linhas)")

        except Exception as e:
            print(f"❌ Falha ao baixar {ticker}: {e}")
            stats[ticker] = {"status": "error", "error": str(e), "inserted": 0, "skipped": 0}

    if not records:
        print("⚠️ Nenhum dado para inserir.")
        return stats

    with engine.begin() as conn:
        stmt = insert(table).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "day_date"],
            set_={k: stmt.excluded[k] for k in ["close", "volume", "log_return"]}
        )
        conn.execute(stmt)

    print(f"💾 Dados inseridos/atualizados para {len(tickers)} tickers.")
    return stats

def get_existing_companies():
    query = text("SELECT DISTINCT ticker FROM companies")
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]


def fetch_and_store_company_info(tickers):
    if isinstance(tickers, str):
        tickers = [tickers]  # 🔥 Converte string em lista

    existing_tickers = set(get_existing_companies())
    tickers_set = set(tickers)

    tickers_to_download = list(tickers_set - existing_tickers)

    if not tickers_to_download:
        print("🚫 Nenhuma empresa nova para inserir.")
        return

    metadata = MetaData()
    table = Table("companies", metadata, autoload_with=engine)

    with engine.begin() as conn:
        for ticker in tickers_to_download:
            try:
                info = yf.Ticker(ticker).info

                if not info :
                    print(f"⚠️ Info não encontrado para {ticker}")
                    continue

                data = {
                    "ticker": ticker.upper(),
                    "long_name": info.get("longName"),
                    "short_name": info.get("shortName"),
                    "currency": info.get("currency"),
                    "exchange": info.get("exchange"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country"),
                    "market_cap": info.get("marketCap"),
                    "website": info.get("website"),
                    "dividend_yield": info.get("dividendYield")
                }

                stmt = insert(table).values(data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker"],
                    set_={key: stmt.excluded[key] for key in data.keys() if key != "ticker"}
                )
                conn.execute(stmt)

                print(f"✅ Info de {ticker} salvo.")

            except Exception as e:
                print(f"⚠️ Erro ao buscar info de {ticker}: {e}")