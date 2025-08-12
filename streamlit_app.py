import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sqlalchemy import text
from scipy import stats
from core.fetchers.fetch_stock import fetch_and_store
from core.database.engine import engine
from arch.unitroot import KPSS
import numpy as np


def get_tickers():
    query = text("SELECT ticker, long_name FROM companies ORDER BY ticker")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        return []
    else:
        return df


@st.cache_data(ttl=300)
def get_data(ticker: str, start_date: pd.Timestamp):
    query = text("""
        SELECT day_date, close, log_return
        FROM stocks
        WHERE ticker = :ticker
          AND day_date >= :start_date
        ORDER BY day_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"ticker": ticker, "start_date": start_date.date()})

@st.cache_data(ttl=300)
def get_company_info(ticker: str):
    query = text("""
        SELECT *
        FROM companies
        WHERE ticker = :ticker
        LIMIT 1
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    return df.iloc[0] if not df.empty else None


@st.cache_data(ttl=300)
def get_sectors():
    q = text("""
        SELECT DISTINCT sector
        FROM companies
        WHERE sector IS NOT NULL
        ORDER BY sector
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn)
    
    
@st.cache_data(ttl=300)
def get_industries_by_sector(sector: str):
    q = text("""
        SELECT DISTINCT industry
        FROM companies
        WHERE sector = :sector
          AND industry IS NOT NULL
        ORDER BY industry
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"sector": sector})
    

@st.cache_data(ttl=300)
def get_tickers_filtered_optional(sector: str = None, industry: str = None):
    base = "SELECT ticker, long_name FROM companies"
    where = []
    params = {}
    if sector:
        where.append("sector = :sector")
        params["sector"] = sector
    if industry:
        where.append("industry = :industry")
        params["industry"] = industry
    if where:
        base += " WHERE " + " AND ".join(where)
    base += " ORDER BY ticker"
    with engine.connect() as conn:
        return pd.read_sql(text(base), conn, params=params)
    
@st.cache_data(ttl=300)
def get_sector_avg_return_sql(sector: str, exclude_ticker: str, start_date: pd.Timestamp):
    query = text("""
        SELECT s.day_date, AVG(s.log_return) AS sector_avg_return
        FROM stocks s
        JOIN companies c ON c.ticker = s.ticker
        WHERE c.sector = :sector
          AND s.ticker <> :exclude
          AND s.day_date >= :start_date
        GROUP BY s.day_date
        ORDER BY s.day_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={
            "sector": sector,
            "exclude": exclude_ticker,
            "start_date": start_date.date()
        })

    


# grafico Basico
def plot_price(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["day_date"],
        y=df["close"],
        mode="lines",
        name=ticker
    ))

    fig.update_layout(
        title=f"Close Price - {ticker}",
        template="seaborn",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis_tickformat="%d-%m-%Y",
        yaxis_tickprefix="$",
        font=dict(color="white"),
        showlegend=True,
        height=500
    )

    return fig

# Cálculo do Drawdown Máximo
def max_drawdown(returns: pd.Series):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()



st.set_page_config(layout="wide")

st.title("Stock Viewer")

# Load available filters
sectors_df = get_sectors()
if sectors_df.empty:
    st.warning("No data in companies table.")
    st.stop()

# Sector filter (optional)
sector = st.sidebar.selectbox(
    "Setor:",
    ["All"] + sectors_df["sector"].dropna().tolist(),
    index=0
)
sector = None if sector == "All" else sector

if sector:
    industries_df = get_industries_by_sector(sector)
    industry_options = ["All"] + industries_df["industry"].dropna().tolist()
else:
    industries_df = pd.DataFrame(columns=["industry"])
    industry_options = ["All"]

industry = st.sidebar.selectbox(
    "Indústria :",
    industry_options,
    index=0,
    disabled=(sector is None)  # só habilita quando tiver setor escolhido
)

industry = None if industry == "All" else industry


# Load tickers based on filters
ticker_df = get_tickers_filtered_optional(sector=sector, industry=industry)
if ticker_df.empty:
    st.warning("No tickers found for this filter.")
    st.stop()

labels = {row['ticker']: f"{row['ticker']} - {row['long_name']}" for _, row in ticker_df.iterrows()}

# Select ticker
ticker = st.selectbox(
    "Select a stock ticker:",
    ticker_df["ticker"],
    format_func=lambda x: labels.get(x, x)
)


col11, col12 = st.columns([1, 1])


with col11:
    lookback_map = {"6 meses": 180, "1 ano": 365, "2 anos": 730, "3 anos": 1095, "5 anos": 1825}
    lookback_option = st.selectbox("Período para análise", list(lookback_map.keys()), index=1)
    lookback_days = lookback_map[lookback_option]
    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    
with col12:
    option = st.radio("Escolha o que exibir:", ("Preço de fechamento", "Retorno acumulado"), horizontal=True)


col21, col22 = st.columns([2, 1])

with col21:
    

    # Carrega dados já no período
    df = get_data(ticker, start_date)

    # Derivados a partir de log_return (sem função externa)
    df = df.sort_values("day_date").copy()
    df["ret"] = np.expm1(df["log_return"].fillna(0.0))  # (e^lr - 1)
    df["cumulative_return"] = (1 + df["ret"]).cumprod() - 1

    if option == "Preço de fechamento":
        st.plotly_chart(plot_price(df, ticker), use_container_width=True, theme="streamlit")
    else:
        st.line_chart(df.set_index("day_date")["cumulative_return"])

with col22:
    st.subheader(f"📈 {ticker} company_info")
    company_info = get_company_info(ticker)

    if company_info is not None:
        with st.expander(f"📊 {company_info['long_name']} ({company_info['ticker']})", expanded=True):
            st.markdown(f"**Setor**: {company_info['sector'] or '—'}")
            st.markdown(f"**Indústria**: {company_info['industry'] or '—'}")
            st.markdown(f"**País**: {company_info['country'] or '—'}")
            st.markdown(f"**Moeda**: {company_info['currency'] or '—'}")
            st.markdown(f"**Bolsa**: {company_info['exchange'] or '—'}")
            st.markdown(f"**Website**: [{company_info['website']}]({company_info['website']})" if company_info['website'] else "—")
            if pd.notna(company_info["market_cap"]):
                st.markdown(f"**Market Cap**: ${company_info['market_cap']:,}")
            if pd.notna(company_info["dividend_yield"]):
                st.markdown(f"**Dividend Yield**: {round(company_info['dividend_yield'], 2)}%")

# Parte 2: Comparação com o Setor
# Parte 2: Comparação com o Setor
st.markdown("---")
st.header("📊 Comparação com o Setor")

def get_sector_avg_return(sector: str, exclude_ticker: str = None):
    query = text("SELECT ticker FROM companies WHERE sector = :sector")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"sector": sector})
    tickers = df["ticker"].tolist()
    if exclude_ticker:
        tickers = [t for t in tickers if t != exclude_ticker]

    dfs = []
    for tkr in tickers:
        df_tmp = get_data(tkr)
        if df_tmp.empty or df_tmp.shape[0] < 100:
            continue
        dfs.append(df_tmp[["day_date", "log_return"]])

    if not dfs:
        return pd.DataFrame()

    all_returns = pd.concat(dfs)
    avg = (
        all_returns
        .groupby("day_date")["log_return"]
        .mean()
        .reset_index()
        .rename(columns={"log_return": "sector_avg_return"})
    )
    return avg

col21, col22 = st.columns([2, 1])
company_info = get_company_info(ticker)

if company_info is not None:
    df_ticker = df[["day_date", "log_return"]].copy()
    sector = company_info["sector"]
    df_sector_avg = get_sector_avg_return_sql(sector, exclude_ticker=ticker, start_date=start_date)

    df_merged = (
        df_ticker.merge(df_sector_avg, on="day_date", how="inner")
                .sort_values("day_date")
                .reset_index(drop=True)
)
    df_merged["cumulative_log_return"] = df_merged["log_return"].cumsum()
    df_merged["sector_cumulative_log_return"] = df_merged["sector_avg_return"].cumsum()

     # Testes Estatísticos
    log_emp = df_merged["log_return"].dropna()
    log_set = df_merged["sector_avg_return"].dropna()

    # Normalidade (sem exibir)
    p_emp = stats.shapiro(log_emp).pvalue
    p_set = stats.shapiro(log_set).pvalue

    is_normal = p_emp > 0.05 and p_set > 0.05

    with col21:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_merged["day_date"], y=df_merged["cumulative_log_return"], name=f"{ticker}"))
        fig.add_trace(go.Scatter(x=df_merged["day_date"], y=df_merged["sector_cumulative_log_return"], name=f"{sector} - média"))
        fig.update_layout(
            title=f"Log Return Acumulado - {ticker} x Média do Setor {sector}",
            xaxis_title="Data",
            yaxis_title="Log Return Acumulado",
            height=500,
            legend=dict(
                x=0.99,
                y=0.99,
                traceorder="normal",
                bgcolor="rgba(0,0,0,0)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        

           
    

        col211, col212 = st.columns(2)

        with col22:
            if is_normal:
                stat, p = stats.ttest_rel(log_emp, log_set)
                st.metric(label="T-Test Pareado", value=f"p = {p:.4f}", delta="Paramétrico")
                if p < 0.05:
                    st.caption("🔍 Diferença significativa nas médias dos log returns.")
                else:
                    st.caption("📊 Os retornos médios do ativo e do setor são semelhantes no período analisado.")
            else:
                stat, p = stats.wilcoxon(log_emp, log_set)
                st.metric(label="Wilcoxon Test", value=f"p = {p:.4f}", delta="Não Paramétrico")
                if p < 0.05:
                    st.caption("🔍 Diferença significativa nas distribuições dos log returns.")
                else:
                    st.caption("📊 O ativo e o setor tiveram retornos similares, sem diferença estatística relevante.")

            stat, p = stats.levene(log_emp, log_set)
            st.metric(label="Teste de Levene", value=f"p = {p:.4f}", delta="Variância")
            if p < 0.05:
                st.caption("⚠️ O ativo apresenta volatilidade significativamente diferente do setor.")
            else:
                st.caption("✅ O ativo e o setor possuem volatilidades semelhantes.")


    # Histograma Suavizado
    hist_data = [
        df_merged["log_return"].dropna(),
        df_merged["sector_avg_return"].dropna()
    ]
    group_labels = [f"{ticker}", f"Sector {sector}"]

    with col21:
        hist_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
        hist_fig.update_layout(title="Distribuição de Log Returns Diários (KDE)")
        st.plotly_chart(hist_fig, use_container_width=True)

    with col22:
        # Análise de Caldas
        st.subheader("📉 Análise de Caudas e Risco Extremo")
        for label, series in zip(group_labels, hist_data):
            left_tail = (series < series.mean() - 2 * series.std()).sum()
            right_tail = (series > series.mean() + 2 * series.std()).sum()
            kurtosis = stats.kurtosis(series)
            skewness = stats.skew(series)

            # EVT
            q05 = np.quantile(series, 0.05)
            q95 = np.quantile(series, 0.95)
            es_left = series[series <= q05].mean()
            es_right = series[series >= q95].mean()

            # Índice de Hill
            sorted_left = np.sort(-series[series < q05])  # valores negativos grandes
            sorted_right = np.sort(series[series > q95])  # valores positivos grandes
        
            if label == f"Sector {sector}":
                max_dd = max_drawdown(df_merged["sector_avg_return"].dropna())
            else:
                max_dd = max_drawdown(df_merged["log_return"].dropna())
            
            
            def hill_estimator(data):
                if len(data) < 2:
                    return np.nan
                logs = np.log(data / data[0])
                return np.mean(logs)

            hill_left = hill_estimator(sorted_left) if len(sorted_left) > 0 else np.nan
            hill_right = hill_estimator(sorted_right) if len(sorted_right) > 0 else np.nan

            st.markdown(f"**{label}**")
            st.write(f"→ Cauda Esquerda (baixa extrema): {left_tail} dias")
            st.write(f"→ Cauda Direita (alta extrema): {right_tail} dias")
            with st.expander("📅 Dias com retornos extremos (fora 2 desvios)"):
                threshold_low = series.mean() - 2 * series.std()
                threshold_high = series.mean() + 2 * series.std()

                if label == f"Sector {sector}":
                    data_filtered = df_merged[["day_date", "sector_avg_return"]].copy()
                    data_filtered["ret"] = data_filtered["sector_avg_return"]
                else:
                    data_filtered = df_merged[["day_date", "log_return"]].copy()
                    data_filtered["ret"] = data_filtered["log_return"]

                extremes = data_filtered[(data_filtered["ret"] < threshold_low) | (data_filtered["ret"] > threshold_high)]
                extremes = extremes.sort_values(by="ret", ascending=True)
                extremes = extremes[["day_date", "ret"]].rename(columns={"ret": "Retorno"})
                extremes["Retorno"] = (extremes["Retorno"] * 100).round(2).astype(str) + "%"

                st.dataframe(extremes.reset_index(drop=True), use_container_width=True)

            st.write(f"→ Curtose: {kurtosis:.2f} ( > 3 indica caudas pesadas)")
            st.write(f"→ Assimetria (Skewness): {skewness:.2f}")
            st.write(f"→ Quantil 5%: {q05* 100:.2f}% | Expected Shortfall: {es_left* 100:.2f}%")
            st.write(f"→ Quantil 95%: {q95*100:.2f}% | Expected Uprasing: {es_right*100:.2f}%")
            st.write(f"→ Índice de Hill (esquerda): {hill_left:.4f} | (direita): {hill_right:.4f}")
            st.write(f"→ Máximo Drawdown no periodo: {max_dd * 100:.2f}%")



    
    # Parte 3: Estatísticas Rolling

    
st.markdown("---")
st.header("📈 Estatísticas Rolling")

rolling_window = st.slider("Escolha a janela de dias para o cálculo rolling", min_value=5, max_value=120, value=30, step=5)



# Cálculo rolling
df_merged["rolling_vol_empresa"] = df_merged["log_return"].rolling(window=rolling_window).std()
df_merged["rolling_vol_setor"] = df_merged["sector_avg_return"].rolling(window=rolling_window).std()
df_merged["rolling_mean_empresa"] = df_merged["log_return"].rolling(window=rolling_window).mean()
df_merged["rolling_sharpe_empresa"] = df_merged["rolling_mean_empresa"] / df_merged["rolling_vol_empresa"]
df_merged["rolling_mean_setor"] = df_merged["sector_avg_return"].rolling(window=rolling_window).mean()
df_merged["rolling_sharpe_setor"] = df_merged["rolling_mean_setor"] / df_merged["rolling_vol_setor"]

# Adiciona o preço (puxa do df original que tem o fechamento)
df_price = df[["day_date", "close"]].set_index("day_date")
df_merged = df_merged.set_index("day_date").join(df_price, how="left").reset_index()

# Cálculo das diferenças
df_merged["diff_sharpe"] = df_merged["rolling_sharpe_empresa"] - df_merged["rolling_sharpe_setor"]
df_merged["diff_vol"] = df_merged["rolling_vol_empresa"] - df_merged["rolling_vol_setor"]

# Função de Z-score rolling
def rolling_zscore(series, window):
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

# Função de detecção de mudanças (CUSUM)
def detect_cusum(x, threshold=0.5):
    s_pos, s_neg = 0, 0
    change_points = []
    for i in range(1, len(x)):
        diff = x.iloc[i] - x.iloc[i-1]
        s_pos = max(0, s_pos + diff)
        s_neg = min(0, s_neg + diff)
        if s_pos > threshold:
            change_points.append(x.index[i])
            s_pos = 0
        elif s_neg < -threshold:
            change_points.append(x.index[i])
            s_neg = 0
    return change_points



# Cálculo da correlação rolling
rolling_corr = (
    df_merged[["log_return", "sector_avg_return"]]
    .rolling(window=rolling_window)
    .corr()
    .unstack()
    .iloc[:,1]
)

df_merged["rolling_corr"] = rolling_corr.values
x = df_merged.set_index("day_date")["rolling_sharpe_empresa"].dropna()
cusum_dates = detect_cusum(x)
# Z-scores das diferenças
df_merged["z_diff_sharpe"] = rolling_zscore(df_merged["diff_sharpe"], rolling_window)
df_merged["z_diff_vol"] = rolling_zscore(df_merged["diff_vol"], rolling_window)

# Flags de anomalias onde z-score > 1.5 (positivo ou negativo)
flag_sharpe = df_merged[abs(df_merged["z_diff_sharpe"]) > 1.92]
flag_vol = df_merged[abs(df_merged["z_diff_vol"]) > 1.92]

shapes = [
    dict(
        type="line",
        xref="x",
        yref="paper",
        x0=date,
        x1=date,
        y0=0,
        y1=1,
        line=dict(color="gray", width=1, dash="dot")
    )
    for date in cusum_dates
]

# Gráfico
layout = dict(
    hoversubplots="axis",
    title=dict(text="Stock Price Changes"),
    hovermode="x",
    shapes=shapes,
    grid=dict(rows=4, columns=1)
)

data = [
    go.Scatter(x=df["day_date"], y=df["close"], xaxis="x", yaxis="y", name="Closed Price"),
    go.Scatter(
    x=df_merged["day_date"],
    y=df_merged["rolling_corr"],
    name="Rolling Corr Empresa-Setor",
    xaxis="x",
    yaxis="y2",
    line=dict(color="green")
    ),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_vol_empresa"], xaxis="x", yaxis="y3", name="Rolling Volatility Empresa"),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_vol_setor"], xaxis="x", yaxis="y3", name="Rolling Volatility Setor"),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_sharpe_setor"], xaxis="x", yaxis="y4", name="Rolling Sharpe Setor"),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_sharpe_empresa"], xaxis="x", yaxis="y4", name="Rolling Sharpe Empresa"),
    go.Scatter(
        x=flag_sharpe["day_date"],
        y=flag_sharpe["rolling_sharpe_empresa"],
        mode="markers",
        marker=dict(color="orange", size=8, symbol="diamond"),
        name="Z-score Sharpe Diff > 1.92",
        xaxis="x",
        yaxis="y4"
    ),
    go.Scatter(
        x=flag_vol["day_date"],
        y=flag_vol["rolling_vol_empresa"],
        mode="markers",
        marker=dict(color="red", size=7, symbol="triangle-up"),
        name="Z-score Vol Diff > 1.92",
        xaxis="x",
        yaxis="y3"
    ),
]




fig = go.Figure(data=data, layout=layout)

fig.update_layout(
    yaxis=dict(domain=[0.75, 1.00], title="Close"),
    yaxis2=dict(domain=[0.50, 0.74], title="Corr Empresa-Setor"),
    yaxis3=dict(domain=[0.25, 0.49], title="Volatility"),
    yaxis4=dict(domain=[0.00, 0.24], title="Sharpe"),
    height=750,
)

fig.add_shape(
    type="line",
    xref="paper",
    yref="y4",  # eixo do Sharpe
    x0=0,
    x1=1,
    y0=0,
    y1=0,
    line=dict(color="black", width=1.5, dash="dash")
)


st.plotly_chart(fig, use_container_width=True)

kpss_sharpe = KPSS(df_merged["rolling_sharpe_empresa"].dropna(), lags=12)
st.metric("KPSS (Sharpe rolling)", f"p = {kpss_sharpe.pvalue:.4f}")
if kpss_sharpe.pvalue < 0.05:
    st.warning("⚠️ Sharpe rolling não estacionário (regime instável)")

print (cusum_dates)
