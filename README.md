# 📈 Trade Analytics

Visualizador de ações com dados históricos armazenados em banco PostgreSQL (Neon) e interface interativa em Streamlit.

## 🚀 Funcionalidades
- Filtro por **setor** e **indústria** (opcionais)
- Visualização de:
  - Preço de fechamento
  - Retorno acumulado
  - Comparação com média do setor
  - Estatísticas de risco e análise de caudas
  - Rolling metrics (volatilidade, Sharpe, correlação)
- Atualização incremental diária dos dados via `yfinance`
- Dados armazenados no banco (evita re-downloads e acelera consultas)

---

## 📸 Capturas de Tela

### Tela Inicial
![Tela inicial](Images/tela_principal.png)

### Comparação com o Setor
![Comparação Setor](Images/comparacao_setor.png)

### Estatísticas Rolling
![Rolling Stats](Images/rolling_metrics.png)

---


## ⚠️ Disclaimer
Este projeto tem caráter **educacional e exploratório**.  
As informações e análises apresentadas **não constituem recomendação de investimento** nem devem ser utilizadas como base exclusiva para decisões financeiras.

O autor **não se responsabiliza** por perdas ou danos decorrentes do uso das informações fornecidas pela aplicação.  
Investir em renda variável envolve riscos, e cada investidor deve fazer sua própria análise ou consultar um profissional habilitado
