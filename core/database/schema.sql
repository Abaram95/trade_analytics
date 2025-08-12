CREATE TABLE IF NOT EXISTS zscore_multiscale (
    ticker TEXT PRIMARY KEY,
    ref_date DATE NOT NULL,
    z_60 FLOAT,
    z_120 FLOAT,
    z_360 FLOAT,
    z_720 FLOAT,
    n_days_available INTEGER,
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    day_date DATE NOT NULL,
    close FLOAT,
    volume BIGINT,
    UNIQUE(ticker, day_date)
);

CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR(10) PRIMARY KEY,
    long_name VARCHAR,
    short_name VARCHAR,
    currency VARCHAR(5),
    exchange VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    country VARCHAR,
    market_cap BIGINT,
    website VARCHAR,
    dividend_yield FLOAT,
    update_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS macro_indicators (
    id SERIAL PRIMARY KEY,
    day_date DATE NOT NULL,
    indicador VARCHAR(100) NOT NULL,
    valor NUMERIC(18,6) NOT NULL,
    unidade VARCHAR(20),
    fonte VARCHAR(50) DEFAULT 'BCB',
    update_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (day_date, indicador)
);

CREATE INDEX IF NOT EXISTS idx_ticker_date ON stocks(ticker, day_date);


