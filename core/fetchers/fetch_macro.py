"""
Descontionado: Este código é parte de um projeto maior e depende de outros módulos e configurações.

Fetcher de dados macroeconômicos via API do Banco Central (SGS).
✔️ Busca todas as séries do dicionário.
✔️ Monta DataFrame consolidado.
✔️ Inclui metadados (unidade, frequência, fonte).
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from sqlalchemy import text
from core.database.engine import engine



# 🔥 Dicionário dos indicadores
INDICATORS = {
    'ipca':                   {'code': 433,    'unit': '%',            'freq': 'monthly'},
    'inpc':                   {'code': 188,    'unit': '%',            'freq': 'monthly'},
    'igp_m':                  {'code': 189,    'unit': '%',            'freq': 'monthly'},
    'igp_di':                 {'code': 190,    'unit': '%',            'freq': 'monthly'},
    'incc':                   {'code': 192,    'unit': '%',            'freq': 'monthly'},
    'selic_anual_252':        {'code': 1178,   'unit': '% a.a.',       'freq': 'daily'},
    'cdi_anual_252':          {'code': 4389,   'unit': '% a.a.',       'freq': 'daily'},
    'tlp':                    {'code': 27572,  'unit': '%',            'freq': 'monthly'},
    'spread_total':           {'code': 20783,  'unit': 'p.p.',         'freq': 'monthly'},
    'spread_pf':              {'code': 20785,  'unit': 'p.p.',         'freq': 'monthly'},
    'spread_pj':              {'code': 20784,  'unit': 'p.p.',         'freq': 'monthly'},
    'taxa_juros_total':       {'code': 20714,  'unit': '% a.a.',       'freq': 'monthly'},
    'taxa_juros_pf':          {'code': 20716,  'unit': '% a.a.',       'freq': 'monthly'},
    'taxa_juros_pj':          {'code': 20715,  'unit': '% a.a.',       'freq': 'monthly'},
    'saldo_credito_total':    {'code': 20539,  'unit': 'R$ milhões',   'freq': 'monthly'},
    'saldo_credito_pj':       {'code': 20540,  'unit': 'R$ milhões',   'freq': 'monthly'},
    'saldo_credito_pf':       {'code': 20541,  'unit': 'R$ milhões',   'freq': 'monthly'},
    'credito_pib_total':      {'code': 20622,  'unit': '% PIB',        'freq': 'monthly'},
    'credito_pib_pj':         {'code': 20623,  'unit': '% PIB',        'freq': 'monthly'},
    'credito_pib_pf':         {'code': 20624,  'unit': '% PIB',        'freq': 'monthly'},
    'endividamento_familias': {'code': 29037,  'unit': '%',            'freq': 'monthly'},
    'comprometimento_renda':  {'code': 29034,  'unit': '%',            'freq': 'monthly'},
    'inadimplencia_total':    {'code': 21082,  'unit': '%',            'freq': 'monthly'},
    'inadimplencia_pf':       {'code': 21084,  'unit': '%',            'freq': 'monthly'},
    'inadimplencia_pj':       {'code': 21083,  'unit': '%',            'freq': 'monthly'},
    'pib_nominal_anual':      {'code': 1207,   'unit': 'R$',           'freq': 'annual'},
    'pib_mensal':             {'code': 4380,   'unit': 'R$ milhões',   'freq': 'monthly'},
    'pib_12m':                {'code': 4382,   'unit': 'R$ milhões',   'freq': 'monthly'},
    'massa_salarial_real':    {'code': 28544,  'unit': 'R$ milhões',   'freq': 'monthly'},
    'rendimento_medio':       {'code': 24382,  'unit': 'R$',           'freq': 'monthly'},
    'dolar':                  {'code': 1,      'unit': 'R$',           'freq': 'daily'},
    'euro':                   {'code': 21619,  'unit': 'R$',           'freq': 'daily'},
    'balanca_comercial':      {'code': 22704,  'unit': 'US$ milhões',  'freq': 'monthly'},
    'reservas_totais':        {'code': 13621,  'unit': 'US$ milhões',  'freq': 'daily'},
    'reservas_liquidez':      {'code': 13982,  'unit': 'US$ milhões',  'freq': 'daily'},
    'divida_bruta_pib':       {'code': 4503,   'unit': '% PIB',        'freq': 'monthly'},
    'divida_liquida_pib':     {'code': 4503,   'unit': '% PIB',        'freq': 'monthly'},
    'confiança_consumidor':   {'code': 4393,   'unit': 'Índice',       'freq': 'monthly'},
    'confiança_industrial':   {'code': 7341,   'unit': 'Índice',       'freq': 'quarterly'},
    'confiança_serviços':     {'code': 17660,  'unit': 'Índice',       'freq': 'monthly'},
    'desemprego_pnad':        {'code': 24369,  'unit': '%',            'freq': 'monthly'},
    'renda_disponivel':       {'code': 29023,  'unit': 'R$ milhões',   'freq': 'monthly'},
}


# 🔥 Fetch individual series
def fetch_bcb_series(code: int, start_date: str = '01/01/2010', end_date: str = '31/12/2025') -> pd.DataFrame:
    start = datetime.strptime(start_date, "%d/%m/%Y").date()
    end = datetime.strptime(end_date, "%d/%m/%Y").date()

    dfs = []

    while start <= end:
        segment_end = min(start + timedelta(days=365 * 10 - 1), end)

        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
            f"?formato=json&dataInicial={start.strftime('%d/%m/%Y')}&dataFinal={segment_end.strftime('%d/%m/%Y')}"
        )

        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

        if r.status_code != 200 or not r.content or r.text.strip() == "":
            start = segment_end + timedelta(days=1)
            continue

        try:
            data = r.json()
        except Exception:
            start = segment_end + timedelta(days=1)
            continue

        df = pd.DataFrame(data)

        if not df.empty:
            df['date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            df['value'] = pd.to_numeric(df['valor'].str.replace(',', '.'), errors='coerce')
            df = df[['date', 'value']].dropna()
            dfs.append(df)

        start = segment_end + timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs).drop_duplicates().sort_values('date')


# 🔥 Fetch all
def fetch_all_macro_to_dataframe(start_date='01/01/2010', end_date='31/12/2025') -> pd.DataFrame:
    dfs = []

    for name, info in INDICATORS.items():
        print(f"🔄 {name}")
        df = fetch_bcb_series(info['code'], start_date, end_date)

        if df.empty:
            continue

        df['indicator'] = name
        df['unit'] = info['unit']
        df['frequency'] = info['freq']
        df['source'] = 'SGS'

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs).drop_duplicates().sort_values(['indicator', 'date']).reset_index(drop=True)


def save_dataframe_to_db(df: pd.DataFrame):
    if df.empty:
        print("DataFrame vazio. Nada foi salvo.")
        return

    with engine.begin() as conn:
        for _, row in df.iterrows():
            stmt = text("""
                INSERT INTO macro_indicators (date, indicator, value, unit, frequency, source)
                VALUES (:date, :indicator, :value, :unit, :frequency, :source)
                ON CONFLICT (date, indicator) DO UPDATE
                SET value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    frequency = EXCLUDED.frequency,
                    source = EXCLUDED.source;
            """)
            conn.execute(stmt, {
                'date': row['date'].date(),
                'indicator': row['indicator'],
                'value': row['value'],
                'unit': row['unit'],
                'frequency': row['frequency'],
                'source': row['source']
            })

    print(f"✅ {len(df)} registros salvos ou atualizados no banco.")


# 🔥 MAIN
if __name__ == "__main__":
    df_macro = fetch_all_macro_to_dataframe()
    print(df_macro.head())
    save_dataframe_to_db(df_macro)

    # 🔥 Modo interativo
    import code
    code.interact(local=locals())
