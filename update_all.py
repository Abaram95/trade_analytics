
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
from core.fetchers.fetch_stock import fetch_and_store  


load_dotenv()
DATABASE_URL = os.getenv("NEONBASE_URL")
engine = create_engine(DATABASE_URL)

def get_all_tickers(engine):
    query = text("SELECT DISTINCT ticker FROM companies")
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
    return [row[0] for row in result]

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def update_all(engine, batch_size=25):
    now = datetime.now()
    date_tag = now.strftime("%Y-%m-%d")
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/update_{date_tag}.log"

    total_inserted = 0
    total_tickers = 0
    total_skipped = 0
    total_failed  = 0

    with open(log_filename, "a", encoding="utf-8") as log:
        def logprint(msg):
            print(msg)
            log.write(msg + "\n")

        logprint(f"Início da atualização: {now.strftime('%Y-%m-%d %H:%M:%S')}")

        tickers = get_all_tickers(engine)
        logprint(f"Encontrados {len(tickers)} tickers.")

        for batch in chunked(tickers, batch_size):
            logprint(f"Atualizando lote ({len(batch)}): {', '.join(batch)}")
            try:
                stats = fetch_and_store(batch)  # <-- passa a lista
                # stats pode ser None se nada foi inserido; trate abaixo
                if stats:
                    for tkr, s in stats.items():
                        total_tickers += 1
                        total_inserted += s.get("inserted", 0)
                        total_skipped  += s.get("skipped", 0)
                        if s.get("status") == "error":
                            total_failed += 1
                            logprint(f"  {tkr}: ERRO: {s.get('error')}")
                        else:
                            logprint(f"  {tkr}: +{s.get('inserted',0)} linhas, "
                                     f"{s.get('skipped',0)} ignoradas "
                                     f"({s.get('start')}→{s.get('end')})")
                else:
                    logprint("  Lote sem novos dados.")
            except Exception as e:
                total_failed += len(batch)
                logprint(f"ERRO no lote: {e}")

        end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logprint(f"Resumo: tickers={total_tickers} | inseridos={total_inserted} | "
                 f"sem-novos={total_skipped} | falhas={total_failed}")
        logprint(f"Atualização concluída em {end_str}.")

if __name__ == "__main__":
    update_all(engine)
