import streamlit as st
import pandas as pd
from datetime import date

from src import data, signals, risk, backtest, evaluation

st.set_page_config(page_title="Alpha + Optimizer Sandbox", layout="wide")
st.title("Alpha + Risk Model Sandbox")

# Inputs
tickers = st.text_input(
    "Tickers (comma-separated)",
    "AAPL,MSFT,NVDA,GOOGL,AMZN,META,JPM,KO,PEP,DIS,ABT"
).split(",")
tickers = [t.strip().upper() for t in tickers if t.strip()]

start = st.date_input("Start", value=pd.to_datetime("2018-01-01")).isoformat()
end = st.date_input("End", value=pd.to_datetime(date.today())).isoformat()

col1, col2, col3, col4 = st.columns(4)
cov_name = col1.selectbox("Covariance", list(risk.COV_FACTORIES.keys()), index=2)
rebal_days = int(col2.number_input("Rebalance (days)", min_value=1, max_value=21, value=5))
risk_aversion = float(col3.number_input("Risk aversion λ", min_value=0.1, max_value=100.0, value=10.0))
l1_turn = float(col4.number_input("L1 turnover γ", min_value=0.0, max_value=10.0, value=1.0))

col5, col6, col7, col8 = st.columns(4)
LB = float(col5.number_input("Lower bound per name", min_value=-0.10, max_value=0.00, value=-0.02, step=0.01, format="%.2f"))
UB = float(col6.number_input("Upper bound per name", min_value=0.00, max_value=0.10, value=0.02, step=0.01, format="%.2f"))
NET = float(col7.selectbox("Net exposure", options=[-1.0, 0.0, 1.0], index=1))
COST_BPS = float(col8.number_input("Cost (bps one-way)", min_value=0.0, max_value=100.0, value=15.0))

if st.button("Run"):
    with st.spinner("Downloading & computing..."):
        px = data.download_prices(tickers, start, end)
        vol = data.download_volume(tickers, start, end)
        adv = data.compute_adv(px, vol, 20)

        mom = signals.momentum_12_2(px)
        rev = signals.short_term_reversal(px, 5)
        pairs = signals.cointegration_pairs_signal(px, max_pairs=50, lookback=252)
        SIG = signals.combine_signals(mom=(mom, 0.5), rev=(rev, 0.3), pairs=(pairs, 0.2))

        cfg = backtest.BacktestConfig(
            rebal_days=rebal_days,
            cov_lookback=252,
            risk_aversion=risk_aversion,
            l1_turnover=l1_turn,
            cost_bps=COST_BPS,
            lb=LB, ub=UB,
            long_only=False,
            net_exposure=NET,
            turnover_cap=0.6,
            liquidity_cap=(0.003 * adv.iloc[-1] / (adv.iloc[-1].sum() + 1e-12)),
        )

        cov_fn = risk.COV_FACTORIES[cov_name]
        res = backtest.run_backtest(px, SIG, cov_fn, cfg)
        summ = evaluation.summarize_strategy(res["pnl"], name=cov_name)

    st.subheader("Summary (net)")
    st.dataframe(summ.to_frame("value"))

    st.subheader("Equity Curve")
    st.line_chart(res["cum"].dropna())

    st.subheader("Turnover")
    st.line_chart(res["turnover"].dropna())

    st.subheader("Predicted vs Realized Vol (rolling)")
    vol_df = evaluation.realized_vs_pred_vol(res["pnl"], res["pred_vol"], window=63)
    st.line_chart(vol_df)
