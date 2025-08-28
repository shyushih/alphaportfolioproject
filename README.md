# Alpha & Portfolio Optimization Project


**TL;DR**: Compare covariance/risk models and run a cost‑aware mean‑variance optimizer with turnover & liquidity controls. Plug any alpha signals (incl. a VECM/pairs module) into the same optimizer. Includes a minimal Streamlit app and a leaderboard script.


## Structure
alpha_portfolio_project/ ├── README.md ├── requirements.txt ├── run_leaderboard.py ├── streamlit_app/ │ └── app.py └── src/ ├── data.py ├── signals.py ├── risk.py ├── optimizer.py ├── backtest.py ├── evaluation.py └── utils.py

## Quickstart
1. Create a virtualenv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt