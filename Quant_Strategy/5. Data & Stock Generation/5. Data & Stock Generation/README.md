
```
Data & Stock Generation
├─ config
│  ├─ settings.py
│  └─ __init__.py
├─ data
│  ├─ master
│  │  └─ master_db.xlsx
│  ├─ pipeline
│  │  ├─ calendar.py
│  │  ├─ fundamental_ratios.py
│  │  ├─ indices_prices.py
│  │  ├─ master_universe.py
│  │  ├─ run_pipeline.py
│  │  ├─ sector_industry.py
│  │  ├─ stocks_prices.py
│  │  └─ __init__.py
│  ├─ processed
│  │  ├─ calendar
│  │  │  ├─ trading_days_future.parquet
│  │  │  ├─ trading_days_past.parquet
│  │  │  └─ trading_days_price.parquet
│  │  └─ fundamental_ratios.parquet
│  └─ raw
│     ├─ BookValue.xlsx
│     ├─ EBITDA.xlsx
│     ├─ EPS.xlsx
│     ├─ Indices.xlsx
│     ├─ Prices.xlsx
│     ├─ RevenuePerShare.xlsx
│     ├─ ROA.xlsx
│     ├─ ROCE.xlsx
│     ├─ ROE.xlsx
│     ├─ tmi.json
│     └─ unique_tickers.xlsx
├─ factors
│  ├─ momentum.py
│  ├─ quality.py
│  ├─ value.py
│  └─ __init__.py
├─ main.py
├─ optimization
│  ├─ optimization.py
│  └─ __init__.py
├─ pyproject.toml
├─ README.md
├─ signals
│  ├─ generator.py
│  └─ __init__.py
└─ uv.lock

```