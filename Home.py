import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sqlalchemy import text
from scipy import stats
from arch.unitroot import KPSS
import numpy as np
import yfinance as yf

from core.database.engine2 import init_db_or_die
from core.helpers.metrics import read_sql_with_metrics

# Inicializa o DB (espera se necessário)

st.set_page_config(layout="wide", page_title="Stock Analytics",initial_sidebar_state="expanded")

TODAY = pd.Timestamp.today().normalize()
PRICE_CACHE_YEARS = 5
PRICE_MIN_DATE = TODAY - pd.Timedelta(days=365 * PRICE_CACHE_YEARS)

def _to_log_return(px: pd.Series) -> pd.Series:
    return np.log(px / px.shift(1))

def _yf_ticker(t: str, country: str | None) -> str:
    t = (t or "").strip()
    if not t:
        return t
    if country and country.lower().startswith("brazil"):
        return t if t.endswith(".SA") else f"{t}.SA"
    return t  # US/EUA e demais ficam puros



def _select_close_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Retorna DF com uma coluna 'close' a partir de Close/Adj Close.
       Suporta colunas simples e MultiIndex do yfinance.
    """
    import pandas as pd

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["close"])

    # 1) Se for MultiIndex (ex.: ('Close','WEGE3.SA'))
    if isinstance(df_raw.columns, pd.MultiIndex):
        # tenta pegar nível 0 = Close / Adj Close
        for label in ("Close", "Adj Close", "close", "adj close", "adj_close"):
            try:
                sub = df_raw.xs(label, axis=1, level=0)  # fatia por nível 0
                # sub pode ser Series (1 ticker) ou DataFrame (n tickers)
                if isinstance(sub, pd.Series):
                    s = sub.rename("close")
                else:
                    # pega a primeira coluna não nula
                    s = sub.ffill().bfill().iloc[:, 0].rename("close")
                return s.to_frame()
            except Exception:
                pass
        # fallback: achata nomes e procura por close/adj_close
        flat_cols = ["_".join(map(str, tup)).lower().replace(" ", "_") for tup in df_raw.columns]
        df_tmp = df_raw.copy()
        df_tmp.columns = flat_cols
        for key in ("close", "adj_close", "adjclose"):
            if key in df_tmp.columns:
                return df_tmp[[key]].rename(columns={key: "close"})

        return pd.DataFrame(columns=["close"])

    # 2) Colunas simples
    cols_map = {c.lower().strip().replace(" ", "_"): c for c in df_raw.columns}
    if "close" in cols_map:
        col = cols_map["close"]
    elif "adj_close" in cols_map or "adjclose" in cols_map:
        col = cols_map.get("adj_close", cols_map.get("adjclose"))
    elif "price" in cols_map:
        col = cols_map["price"]
    else:
        return pd.DataFrame(columns=["close"])
    return df_raw[[col]].rename(columns={col: "close"}).copy()



# Carrega e cacheia o dataframe de companies (usado para labels)
@st.cache_data(ttl=3600, show_spinner=False)
def load_companies_df():
    q = text("""
        SELECT *
        FROM companies   -- era companies_pt
        ORDER BY ticker
    """)
    df = read_sql_with_metrics(q, endpoint="load_companies_df")
    for col in ("sector", "industry", "country"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_returns_5y(sector: str, country: str | None) -> pd.DataFrame:
    if not sector or sector == "All":
        return pd.DataFrame(columns=["day_date", "sector_avg_return"])

    mask = (companies_df["sector"] == sector)
    if country and country != "All":
        mask &= (companies_df["country"] == country)

    tickers = (companies_df.loc[mask, "ticker"].dropna().astype(str).unique().tolist())
    if not tickers:
        return pd.DataFrame(columns=["day_date", "sector_avg_return"])

    start = PRICE_MIN_DATE.date()
    frames = []
    for t in tickers:
        # país específico de cada ticker
        ctry = companies_df.loc[companies_df["ticker"] == t, "country"].head(1)
        ctry = ctry.iloc[0] if not ctry.empty else None
        yf_t = _yf_ticker(t, ctry)

        d = yf.download(yf_t, start=start, progress=False, auto_adjust=False, group_by="column", threads=True)
        if d is None or d.empty:
            continue
        px = _select_close_column(d)
        if px.empty:
            continue
        frames.append(px.rename(columns={"close": yf_t})[yf_t])

    if not frames:
        return pd.DataFrame(columns=["day_date", "sector_avg_return"])

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    lr = np.log(prices / prices.shift(1))
    sector_avg = lr.mean(axis=1, skipna=True).rename("sector_avg_return")

    out = sector_avg.dropna().to_frame()
    out["day_date"] = out.index
    return out.reset_index(drop=True).sort_values("day_date")[["day_date", "sector_avg_return"]]



# Funções de acesso ao DB com cache
# @st.cache_data(ttl=3600)
# def get_sector_returns_5y(sector: str) -> pd.DataFrame:
#     if not sector:
#         return pd.DataFrame(columns=["day_date","sector_avg_return"])
#     q = text("""
#         SELECT day_date, avg_log_return AS sector_avg_return
#         FROM sector_daily_returns_pt
#         WHERE sector = :s AND day_date >= :d
#         ORDER BY day_date
#     """)
#     df = read_sql_with_metrics(q, params={"s": sector, "d": PRICE_MIN_DATE.date()}, endpoint="get_sector_returns_5y")
#     df["day_date"] = pd.to_datetime(df["day_date"]).dt.tz_localize(None)
#     return df.sort_values("day_date").reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history_5y(ticker: str) -> pd.DataFrame:
    if not ticker:
        return pd.DataFrame(columns=["day_date", "close", "log_return"])

    # país do ticker
    row = companies_df.loc[companies_df["ticker"] == ticker].head(1)
    country = (row["country"].iloc[0] if not row.empty else None)

    yf_t = _yf_ticker(ticker, country)
    df = yf.download(
        yf_t, start=PRICE_MIN_DATE.date(),
        progress=False, auto_adjust=False, group_by="column", threads=True
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["day_date", "close", "log_return"])

    df = _select_close_column(df)
    if df.empty:
        return pd.DataFrame(columns=["day_date", "close", "log_return"])

    df["day_date"] = pd.to_datetime(df.index).tz_localize(None)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df[["day_date", "close", "log_return"]]



# @st.cache_data(ttl=3600)
# def get_price_history_5y(ticker: str) -> pd.DataFrame:
#     q = text("""
#         SELECT day_date, close, log_return
#         FROM stocks
#         WHERE ticker = :t AND day_date >= :d
#         ORDER BY day_date
#     """)
#     df = read_sql_with_metrics(q, params={"t": ticker, "d": PRICE_MIN_DATE.date()},endpoint="get_price_history_5y")
#     # tipos consistentes
#     df["day_date"] = pd.to_datetime(df["day_date"]).dt.tz_localize(None)
#     df = df.sort_values("day_date").reset_index(drop=True)
#     return df

# Cálculo do Drawdown Máximo
def max_drawdown(returns: pd.Series):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


# grafico Basico
def plot_price(df, applied_ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["day_date"],
        y=df["close"],
        mode="lines",
        name=applied_ticker
    ))

    fig.update_layout(
        title=f"{applied_ticker}",
        template="seaborn",
        xaxis_title="Data",
        yaxis_title="Preço de Fechamento (R$)",
        xaxis_tickformat="%d-%m-%Y",
        yaxis_tickprefix="$",
        font=dict(color="white"),
        showlegend=True,
        height=500
    )

    return fig

def _on_sector_change():
    st.session_state["flt_industry"] = "All"
    st.session_state["applied"]["ticker"] = None
    st.session_state["pending_ticker"] = None

def _on_industry_change():
    st.session_state["applied"]["ticker"] = None
    st.session_state["pending_ticker"] = None

def init_state():
    ss = st.session_state
    if "applied" not in ss or not isinstance(ss.get("applied"), dict):
        ss["applied"] = {"ticker": None, "sector": None, "industry": None}
    ss.setdefault("flt_sector", "All")
    ss.setdefault("flt_industry", "All")
    ss.setdefault("pending_ticker", None)


# Começo do Codigo do app



# st.set_page_config(layout="wide")

# st.title("Stock Analytics")

# companies_df = load_companies_df()
# if companies_df.empty:
#     st.warning("No data in companies table.")
#     st.stop()

# colA, colB = st.columns([3, 1], border=True, gap="small")




# with colB:
#     st.subheader("Filtros")
#     sector_opts = ["All"] + sorted(companies_df["sector"].dropna().unique().tolist())
#     sector_sel = st.selectbox("Setor", sector_opts, index=0, key="flt_sector")

#     sector_sel = None if sector_sel == "All" else sector_sel

#     if sector_sel:
#         industry_opts = ["All"] + sorted(companies_df.loc[companies_df["sector"] == sector_sel, "industry"].dropna().unique().tolist())
#     else:
#         industry_opts = ["All"]

#     industry_sel = st.selectbox("Indústria", industry_opts, index=0,disabled=(sector_sel is None), key="flt_industry")


#     # Aplica filtros em memória
#     sector_val = None if sector_sel == "All" else sector_sel
#     industry_val = None if industry_sel == "All" else industry_sel

#     mask = pd.Series(True, index=companies_df.index)
#     if sector_val:
#         mask &= companies_df["sector"] == sector_val
#     if industry_val:
#         mask &= companies_df["industry"] == industry_val

#     ticker_df = (
#     companies_df.loc[mask, ["ticker", "long_name"]]
#     .dropna(subset=["ticker"]).sort_values("ticker")
#     )
#     if ticker_df.empty:
#         st.warning("No tickers found for this filter.")
#         st.stop()

#     labels = {r["ticker"]: f'{r["ticker"]} - {r["long_name"]}' for _, r in ticker_df.iterrows()}

# st.session_state.setdefault("applied", {})
# st.session_state["applied"].setdefault("ticker", None)

# opts = ticker_df["ticker"].tolist()
# labels = {r.ticker: f"{r.ticker} - {r.long_name}" for r in ticker_df.itertuples()}

# applied_ticker = st.session_state["applied"]["ticker"]
# if applied_ticker in opts:
#     st.session_state["flt_ticker"] = applied_ticker
# elif st.session_state.get("flt_ticker") not in opts:
#     st.session_state["flt_ticker"] = opts[0] if opts else None


# # estado aplicado (primeira vez: vazio)
# with colA:
#     st.subheader("Seleção de Ação")
#     opts = ticker_df["ticker"].tolist()

#     applied = st.session_state.get("applied", {}).get("ticker")
#     if applied in opts:
#         st.session_state["flt_ticker"] = applied
#     elif "flt_ticker" not in st.session_state or st.session_state["flt_ticker"] not in opts:
#         st.session_state["flt_ticker"] = opts[0]

#     with st.form("Ticker Selection", clear_on_submit=False, border=False):
#         ticker = st.selectbox(
#             "Escolha uma ação:",
#             opts,
#             format_func=lambda x: labels.get(x, x),
#             key="flt_ticker",
#         )
#         submitted = st.form_submit_button("Analisar")

# if submitted:
#     st.session_state["applied"]["ticker"] = st.session_state["flt_ticker"]
#     applied_ticker = st.session_state["applied"]["ticker"]

# if not applied_ticker:
#     st.info("Selecione um ticker e clique **Analisar**.")
#     st.stop()

# company_info = companies_df.loc[companies_df["ticker"] == applied_ticker].squeeze()
# sector_name  = company_info["sector"] if isinstance(company_info, pd.Series) else None

# df_full = get_price_history_5y(applied_ticker)
# df_s = get_sector_returns_5y(sector_name)
init_db_or_die()
init_state()

st.set_page_config(layout="wide")
st.title("Stock Analytics")

companies_df = load_companies_df()
if companies_df.empty:
    st.warning("No data in companies table.")
    st.stop()

colA, colB = st.columns([3, 1], border=True, gap="small")

with colB:
    st.subheader("Filtros")

    country_opts = ["All"] + sorted(companies_df["country"].dropna().unique().tolist())
    country_sel = st.selectbox("País", country_opts, key="flt_country")

    # setores disponíveis do país selecionado
    mask_country = (companies_df["country"] == country_sel) if country_sel != "All" else pd.Series(True, index=companies_df.index)
    sector_opts = ["All"] + sorted(companies_df.loc[mask_country, "sector"].dropna().unique().tolist())
    sector_sel = st.selectbox("Setor", sector_opts, key="flt_sector", on_change=_on_sector_change)

    if sector_sel != "All":
        mask_ind = mask_country & (companies_df["sector"] == sector_sel)
        industry_opts = ["All"] + sorted(companies_df.loc[mask_ind, "industry"].dropna().unique().tolist())
        ind_disabled = False
    else:
        industry_opts = ["All"]
        ind_disabled = True

    industry_sel = st.selectbox("Indústria", industry_opts, key="flt_industry", disabled=ind_disabled, on_change=_on_industry_change)

    # aplica filtros em memória (inclui país)
    mask = mask_country.copy()
    if sector_sel != "All":
        mask &= companies_df["sector"] == sector_sel
    if industry_sel != "All":
        mask &= companies_df["industry"] == industry_sel
    ticker_df = (
        companies_df.loc[mask, ["ticker", "long_name"]]
        .dropna(subset=["ticker"]).sort_values("ticker")
    )

    if ticker_df.empty:
        st.warning("No tickers found for this filter.")
        st.stop()

    labels = {r.ticker: f"{r.ticker} - {r.long_name}" for r in ticker_df.itertuples()}


# ----------------------------
# Seleção de ticker (com form)
# ----------------------------
with colA:
    st.subheader("Seleção de Ação")
    opts = ticker_df["ticker"].tolist()
    ss = st.session_state
    applied_ticker = ss["applied"].get("ticker")

    if ss["pending_ticker"] not in opts:
        if applied_ticker in opts:
            ss["pending_ticker"] = applied_ticker
        else:
            ss["pending_ticker"] = opts[0] if opts else None

    with st.form("Ticker Selection", clear_on_submit=False, border=False):
        st.selectbox(
            "Escolha uma ação:",
            opts,
            format_func=lambda x: labels.get(x, x),
            key="pending_ticker",
        )
        submitted = st.form_submit_button("Analisar")

if submitted:
    ss["applied"]["ticker"]   = ss["pending_ticker"]
    ss["applied"]["sector"]   = sector_sel
    ss["applied"]["industry"] = industry_sel
    ss["applied"]["country"]  = country_sel



applied_ticker = ss["applied"].get("ticker")
applied_country = ss["applied"].get("country")



if not applied_ticker:
    st.info("Selecione um ticker e clique **Analisar**.")
    st.stop()

company_row = companies_df.loc[companies_df["ticker"] == applied_ticker]

if company_row.empty:
    st.error("Ticker não encontrado na base após aplicar filtros.")
    st.stop()

sector_name  = company_row.squeeze().get("sector")
country_name = company_row.squeeze().get("country")  # garante país real do ticker

with st.spinner("Carregando dados..."):
    df_full = get_price_history_5y(applied_ticker)
    df_s    = get_sector_returns_5y(sector_name, country_name)  # <- setor do MESMO país


# Gate: exige um ticker aplicado
if not applied_ticker:
    st.info("Selecione um ticker e clique **Analisar**.")
    st.stop()

# ----------------------------
# Carrega dados do ticker/segmento
# ----------------------------
company_row = companies_df.loc[companies_df["ticker"] == applied_ticker]
if company_row.empty:
    st.error("Ticker não encontrado na base após aplicar filtros.")
    st.stop()

sector_name = company_row.squeeze().get("sector")

with st.spinner("Carregando dados..."):
    df_full = get_price_history_5y(applied_ticker)
    df_s = get_sector_returns_5y(sector_name, country_name)  # <- setor do MESMO país


# Datas do lookback (fatia em memória sobre os 10 anos em cache)



col11, col12 = st.columns([1, 1])


with col11:
    lookback_map = {"6 meses": 180, "1 ano": 365, "3 anos": 1095, "5 anos": 1825}
    lookback_option = st.radio("Período para análise", list(lookback_map.keys()), index=1, horizontal=True)
    lookback_days = lookback_map[lookback_option]
    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days) 

for _d in [df_full, df_s]:
    _d["day_date"] = pd.to_datetime(_d["day_date"]).dt.tz_localize(None)

df= df_full.loc[df_full["day_date"] >= start_date, ["day_date","close","log_return"]].copy()
df_sector_avg = df_s.loc[df_s["day_date"] >= start_date, ["day_date","sector_avg_return"]].copy()


col21, col22 = st.columns([2, 1])



with col21:
    
    st.plotly_chart(plot_price(df, applied_ticker), use_container_width=True, theme="streamlit")




with col22:
    st.subheader(f"Informaçoes da Companhia")
    company_info = companies_df.loc[companies_df["ticker"] == applied_ticker].squeeze()

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
st.markdown("---")
st.header("📊 Comparação com o Setor")


col21, col22 = st.columns([2, 1])



df_merged = (
    df[["day_date", "log_return"]]
    .merge(df_sector_avg, on="day_date", how="inner")
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

    # ===== Testes Estatísticos (pareados corretamente) =====
pair = (
    df_merged[["day_date", "log_return", "sector_avg_return"]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["log_return", "sector_avg_return"])
    .sort_values("day_date")
)

log_emp = pair["log_return"].to_numpy()
log_set = pair["sector_avg_return"].to_numpy()
n_pairs = len(pair)

with col22:
    st.caption(f"Observações pareadas após limpeza: {n_pairs}")

if n_pairs < 3:
    with col22:
        st.warning("Poucos pares após alinhamento/limpeza (< 3). Pulo os testes estatísticos.")
else:
    n_shapiro = min(n_pairs, 5000)
    p_emp = stats.shapiro(log_emp[:n_shapiro]).pvalue
    p_set = stats.shapiro(log_set[:n_shapiro]).pvalue
    is_normal = (p_emp > 0.05) and (p_set > 0.05)

with col21:
    COLORS = {"empresa": "#1f77b4", "setor": "#ff8316"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_merged["day_date"], y=df_merged["cumulative_log_return"], name=f"{applied_ticker}",mode='lines', line=dict(color=COLORS["empresa"])))
    fig.add_trace(go.Scatter(x=df_merged["day_date"], y=df_merged["sector_cumulative_log_return"], name=f"Sector - {sector_name}",mode='lines', line=dict(color=COLORS["setor"])))
    fig.update_layout(
        title=f"Log Return Acumulado - {applied_ticker} x Média do Setor '{sector_name}' ({country_name})",
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


with col21:
    colA, colB = st.columns(2)
    with colA:
        if is_normal:
            stat, p = stats.ttest_rel(log_emp, log_set)
            st.metric(label="T-Test Pareado", value=f"p = {p:.4f}", delta="Paramétrico")
            if p < 0.05:
                st.caption("🔍 Diferença significativa nas médias dos Retornos Log.")
            else:
                st.caption("📊 Os retornos médios do ativo e do setor são semelhantes no período analisado.")
        else:
            stat, p = stats.wilcoxon(log_emp, log_set)
            help_wilcoxon_md = """\
                **Wilcoxon**
                - Mede se a mediana das diferenças (ativo − setor) é igual a 0.
                - **p ≥ 0,05**: O ativo e o setor são similares, podemos comparar os retornos
                - **p < 0,05**: Não devemos comparar o ativo com o setor, são diferentes
                """
            decision_ok = p >= 0.05
            delta_num = 1 if decision_ok else -1
            delta_txt = "Usar setor como benchmark" if decision_ok else "Não usar setor como referência comparativa"
            st.metric(
                label="Teste Wilcoxon",
                value=f"p = {p:.4f}",
                delta=delta_txt,
                delta_color="normal" if decision_ok else "inverse",
                help=help_wilcoxon_md,
            )

            if p < 0.05:
                st.caption("🔍 Diferença significativa nas distribuições dos Retornos Log.")
            else:
                st.caption("📊 O ativo e o setor tiveram retornos similares, sem diferença estatística relevante.")
    with colB:
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
    group_labels = [f"{applied_ticker}", f"Sector {sector_name}"]

    with col21:
        hist_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
        hist_fig.update_layout(title="Distribuição de Retornos Log Diários (KDE)")
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
        
            if label == f"Sector {sector_name}":
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

                if label == f"Sector {sector_name}":
                    data_filtered = df_merged[["day_date", "sector_avg_return"]].copy()
                    data_filtered["ret"] = data_filtered["sector_avg_return"]
                else:
                    data_filtered = df_merged[["day_date", "log_return"]].copy()
                    data_filtered["ret"] = data_filtered["log_return"]

                extremes = data_filtered[(data_filtered["ret"] < threshold_low) | (data_filtered["ret"] > threshold_high)]
                extremes = extremes.sort_values(by="ret", ascending=True)
                extremes = extremes[["day_date", "ret"]].rename(columns={"ret": "Retorno","day_date": "Data"})
                extremes["Data"] = extremes["Data"].dt.strftime("%d-%m-%Y")
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
st.header("📈 Estatísticas Móveis")

rolling_window = st.slider("Escolha a janela de dias para o cálculo das estatísticas moveis", min_value=5, max_value=120, value=30, step=5)



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
    hovermode="x",
    shapes=shapes,
    grid=dict(rows=4, columns=1)
)

data = [
    go.Scatter(x=df["day_date"], y=df["close"], xaxis="x", yaxis="y", name="Preço de Fechamento"),
    go.Scatter(
    x=df_merged["day_date"],
    y=df_merged["rolling_corr"],
    name="Correlação Móvel Setor-Empresa",
    xaxis="x",
    yaxis="y2",
    line=dict(color="green")
    ),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_vol_empresa"], xaxis="x", yaxis="y3", name="Volatilidade Móvel Empresa"),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_vol_setor"], xaxis="x", yaxis="y3", name="Volatilidade Móvel Setor"),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_sharpe_setor"], xaxis="x", yaxis="y4", name="Sharpe Móvel Setor"),
    go.Scatter(x=df_merged["day_date"], y=df_merged["rolling_sharpe_empresa"], xaxis="x", yaxis="y4", name="Sharpe Móvel Empresa"),
    go.Scatter(
        x=flag_sharpe["day_date"],
        y=flag_sharpe["rolling_sharpe_empresa"],
        mode="markers",
        marker=dict(color="orange", size=8, symbol="diamond"),
        name="Ponto de Desvio Sharpe",
        xaxis="x",
        yaxis="y4"
    ),
    go.Scatter(
        x=flag_vol["day_date"],
        y=flag_vol["rolling_vol_empresa"],
        mode="markers",
        marker=dict(color="red", size=7, symbol="triangle-up"),
        name="Ponto de Desvio Volatilidade",
        xaxis="x",
        yaxis="y3"
    ),
]




fig = go.Figure(data=data, layout=layout)

fig.update_layout(
    yaxis=dict(domain=[0.75, 1.00], title="Preço Fechamento"),
    yaxis2=dict(domain=[0.50, 0.74], title="Correlação"),
    yaxis3=dict(domain=[0.25, 0.49], title="Volatilidade"),
    yaxis4=dict(domain=[0.00, 0.24], title="Índice Sharpe"),
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
    line=dict(color="white", width=1.7, dash="solid")
)


st.plotly_chart(fig, use_container_width=True)

kpss_sharpe = KPSS(df_merged["rolling_sharpe_empresa"].dropna(), lags=12)
st.metric("KPSS (Sharpe rolling)", f"p = {kpss_sharpe.pvalue:.4f}")
if kpss_sharpe.pvalue < 0.05:
    st.warning("⚠️ Sharpe rolling não estacionário, não podemos confiar em médias históricas.")
