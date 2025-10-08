import time
import pandas as pd
from sqlalchemy import text
from core.database.engine2 import engine

def read_sql_with_metrics(query: str, params=None, endpoint: str = "unknown") -> pd.DataFrame:
    t0 = time.perf_counter()
    ok = True
    err = None
    df = pd.DataFrame()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params or {})
    except Exception as e:
        ok = False
        err = str(e)[:500]
        raise
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        approx_bytes = int(df.memory_usage(deep=True).sum()) if not df.empty else 0
        with engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO app_query_metrics(endpoint, rows, ms, approx_bytes, ok, error)
                        VALUES (:endpoint,:rows,:ms,:bytes,:ok,:err)"""),
                dict(endpoint=endpoint, rows=int(len(df)), ms=ms, bytes=approx_bytes, ok=ok, err=err)
            )
    return df