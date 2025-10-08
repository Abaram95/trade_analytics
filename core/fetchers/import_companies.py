import os
import time
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, List

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
# ------------ Nasdaq-100 (oficial Nasdaq.com) ------------
NDX_API_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100?assetclass=stocks"

def get_nasdaq100_symbols(timeout: int = 20) -> list[str]:
    """Lê a lista do Nasdaq-100 direto da API pública da Nasdaq."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
        "Cache-Control": "no-cache",
    }
    r = requests.get(NDX_API_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # A API às vezes muda a chave; tentamos caminhos comuns
    rows = None
    if isinstance(data, dict):
        d1 = data.get("data") or {}
        # já vi tanto "rows" quanto "data"
        rows = d1.get("rows") or d1.get("data") or []
        if isinstance(rows, dict) and "rows" in rows:
            rows = rows["rows"]

    if not isinstance(rows, list):
        raise RuntimeError("Resposta inesperada da API do Nasdaq-100.")

    syms = []
    for r in rows:
        sym = (r.get("symbol") or r.get("Symbol") or "").strip()
        if sym:
            # yfinance prefere hífen em tickers com ponto (ex.: BRK.B -> BRK-B)
            syms.append(sym.upper().replace(".", "-"))
    if not syms:
        raise RuntimeError("Não encontrei símbolos na API do Nasdaq-100.")
    return syms






load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# ------------ Map YF -> DB ------------
_YF2DB = {
    "longName":      "long_name",
    "shortName":     "short_name",
    "currency":      "currency",
    "exchange":      "exchange",
    "sector":        "sector",
    "industry":      "industry",
    "country":       "country",
    "marketCap":     "market_cap",
    "website":       "website",
    "dividendYield": "dividend_yield",
}

def _clean_value(k: str, v: Any) -> Any:
    if v is None:
        return None
    if k == "market_cap":
        try:
            return int(v)
        except Exception:
            return None
    if k == "dividend_yield":
        try:
            v = float(v)
            return v * 100 if v < 1 else v  # fração -> %
        except Exception:
            return None
    if isinstance(v, (list, dict)):
        return None
    return v

# ------------ Coleta de tickers (S&P500) ------------
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_sp500_symbols(timeout: int = 20) -> List[str]:
    """Tenta por id=constituents; se falhar, cai pra 'wikitable'; se falhar, pd.read_html."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36"
    }
    r = requests.get(WIKI_SP500_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable"})
    symbols: List[str] = []

    if table:
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if not tds:
                continue
            a = tds[0].find("a")
            sym = (a.get_text(strip=True) if a else tds[0].get_text(strip=True)).upper()
            if sym:
                symbols.append(sym.replace(".", "-"))  # opcional (BRK.B -> BRK-B)
        if symbols:
            return symbols

    # Fallback robusto
    for tb in pd.read_html(html):
        cols = [str(c).strip().lower() for c in tb.columns]
        if "symbol" in cols:
            col = tb.columns[cols.index("symbol")]
            syms = (tb[col].astype(str).str.strip().str.upper().str.replace(r"\.", "-", regex=True).tolist())
            if syms:
                return syms

    raise RuntimeError("Falha ao extrair tickers do S&P500.")

# ------------ yfinance -> row ------------
def fetch_company_meta(ticker: str) -> Dict[str, Any]:
    tkr = yf.Ticker(ticker)
    try:
        info = tkr.get_info() if hasattr(tkr, "get_info") else tkr.info
    except Exception:
        info = {}

    row = {
        "ticker": ticker,
        "update_at": datetime.now(timezone.utc).astimezone().replace(tzinfo=None),
    }
    for yf_key, db_col in _YF2DB.items():
        row[db_col] = _clean_value(db_col, info.get(yf_key))
    return row

def fetch_many(tickers: Iterable[str], batch: int = 32, delay: float = 0.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    uniq = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))
    for i in range(0, len(uniq), batch):
        for t in uniq[i:i+batch]:
            out.append(fetch_company_meta(t))
        if delay > 0:
            time.sleep(delay)
    return out

# ------------ UPSERT ------------
_UPSERT_SQL = """
INSERT INTO companies
    (ticker, long_name, short_name, currency, exchange, sector, industry, country,
     market_cap, website, dividend_yield, update_at)
VALUES
    (:ticker, :long_name, :short_name, :currency, :exchange, :sector, :industry, :country,
     :market_cap, :website, :dividend_yield, :update_at)
ON CONFLICT (ticker) DO UPDATE SET
    long_name       = EXCLUDED.long_name,
    short_name      = EXCLUDED.short_name,
    currency        = EXCLUDED.currency,
    exchange        = EXCLUDED.exchange,
    sector          = EXCLUDED.sector,
    industry        = EXCLUDED.industry,
    country         = EXCLUDED.country,
    market_cap      = EXCLUDED.market_cap,
    website         = EXCLUDED.website,
    dividend_yield  = EXCLUDED.dividend_yield,
    update_at       = EXCLUDED.update_at
"""

def upsert_companies(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    n = 0
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text(_UPSERT_SQL), r)
            n += 1
    return {"upserted": n}

# ------------ Main ------------
if __name__ == "__main__":
    try:
        ndx = get_nasdaq100_symbols()
    except Exception as e:
        raise SystemExit(f"Falhou ao obter Nasdaq-100 oficial: {e}")

    tickers = sorted(set(ndx))
    print(f"{len(tickers)} tickers para processar (Nasdaq-100)")

    rows = fetch_many(tickers, batch=32, delay=0.0)
    stats = upsert_companies(rows)
    print(f"Upsert concluído: {stats}")


