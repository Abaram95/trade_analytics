from sqlalchemy import text
from engine2 import engine

def delete_tickers(tickers: list[str]):
    """
    Deleta todos os registros de uma lista de tickers das tabelas `stocks` e `companies`.
    """
    if not tickers:
        print("Nenhum ticker informado.")
        return

    # Normaliza para upper e remove duplicatas
    tickers = list({t.strip().upper() for t in tickers if t.strip()})
    if not tickers:
        print("Lista de tickers vazia após limpeza.")
        return

    try:
        with engine.connect() as conn:
            results = {}
            for t in tickers:
                stock_count = conn.execute(
                    text("SELECT COUNT(*) FROM stocks WHERE ticker = :ticker"),
                    {"ticker": t}
                ).scalar()
                company_count = conn.execute(
                    text("SELECT COUNT(*) FROM companies WHERE ticker = :ticker"),
                    {"ticker": t}
                ).scalar()
                results[t] = (stock_count, company_count)

        print("\nResumo dos tickers encontrados:")
        for t, (sc, cc) in results.items():
            print(f"  {t}: {sc} registros em 'stocks', {cc} em 'companies'")

        confirm = input(f"\nConfirmar deleção de {len(tickers)} tickers? [Y/N]: ").strip().lower()
        if confirm != "y":
            print("Operação cancelada.")
            return

        # Deleta todos dentro de uma transação
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM stocks WHERE ticker = ANY(:tickers)"), {"tickers": tickers})
            conn.execute(text("DELETE FROM companies WHERE ticker = ANY(:tickers)"), {"tickers": tickers})

        print(f"\nTickers {tickers} deletados com sucesso.")
    except Exception as e:
        print(f"Erro ao deletar tickers {tickers}: {e}")


if __name__ == "__main__":
    raw = input("Digite os tickers a serem deletados (separados por vírgula): ")
    tickers = [t.strip() for t in raw.split(",") if t.strip()]
    delete_tickers(tickers)
