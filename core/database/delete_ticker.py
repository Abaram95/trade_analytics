from sqlalchemy import text
from core.database.engine import engine

def delete_ticker(ticker: str):
    """
    Deleta todos os registros de um ticker das tabelas `stocks` e `companies`.
    """
    with engine.connect() as conn:
        try:
            stock_count = conn.execute(text("SELECT COUNT(*) FROM stocks WHERE ticker = :ticker"), {"ticker": ticker}).scalar()
            company_count = conn.execute(text("SELECT COUNT(*) FROM companies WHERE ticker = :ticker"), {"ticker": ticker}).scalar()

            if stock_count == 0 and company_count == 0:
                print(f"Nenhum registro encontrado para o ticker '{ticker}'. Nada a deletar.")
                return

            print(f"\nForam encontrados:")
            print(f"  → {stock_count} registros em 'stocks'")
            print(f"  → {company_count} registros em 'companies'\n")

            confirm = input(f"Confirmar deleção de '{ticker}'? [Y/N]: ").strip().lower()
            if confirm != 'y':
                print("Operação cancelada.")
                return

            with engine.begin() as conn:
                conn.execute(text("DELETE FROM stocks WHERE ticker = :ticker"), {"ticker": ticker})
                conn.execute(text("DELETE FROM companies WHERE ticker = :ticker"), {"ticker": ticker})

            print(f"Ticker '{ticker}' deletado com sucesso de ambas as tabelas.")
        except Exception as e:
            print(f"Erro ao deletar ticker '{ticker}':", e)

if __name__ == "__main__":
    ticker = input("Digite o ticker a ser deletado: ").strip().upper()
    delete_ticker(ticker)
