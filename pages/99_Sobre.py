import streamlit as st

st.set_page_config(layout="wide", page_title="Sobre")
st.title("Sobre o projeto")

st.markdown("""
## Motivação
Este projeto nasce da hipótese comum em Valuation de que uma parte relevante do retorno das empresas é explicada pelo **setor em que estão inseridas**. Em outras palavras: o “solo” (setor) costuma pesar mais que a “semente” (empresa). Para testar isso, comparo cada empresa com a **média de retornos do seu setor**, observando convergências e desvios persistentes.

## O que o dashboard mostra
1) **Preço & dados da empresa**  
   Gráfico de preço (candlestick) e informações básicas.

2) **Comparativo com o setor**  
   - Série de retornos da empresa vs. média do setor.  
   - Eventos de **quebra de padrão** (empresa se afasta da média).  
   - **Wilcoxon** para similaridade de distribuições e **teste de variância** (apenas conferência).

3) **Risco e comportamento**  
   - **Max drawdown diário** e **do período**.  
   - **Índice de tendência (hill_index)** para empresa e setor.

4) **Estatísticas móveis (rolling)**  
   - **Correlação empresa×setor** no tempo (acoplamentos/defasagens).  
   - **Volatilidade móvel** (setor tende a ser menos volátil).  
   - **Índice de Sharpe móvel**.

### Fórmula do Índice de Sharpe """
)

st.latex(r"\text{Sharpe}_t = \frac{\mu_t - R_f}{\sigma_t}")
st.caption("μ_t = retorno médio da janela; σ_t = desvio-padrão dos retornos; R_f = taxa livre de risco (0 se não parametrizado).")



st.markdown("""## Limites e cuidados
- Sensível à **janela** e a **defasagens** entre empresa e setor.  
- **Não é recomendação de investimento**.

## Contato
Autor: Marco Antonio Fraga — <marcoantoniocffraga@gmail.com>
""")
