from __future__ import annotations


os.makedirs("outputs", exist_ok=True)


print("Downloading data...")
px = data.download_prices(TICKERS, START, END)
vol = data.download_volume(TICKERS, START, END)
adv = data.compute_adv(px, vol, window=20)


# Signals
print("Computing signals...")
sig_mom = signals.momentum_12_2(px)
sig_rev = signals.short_term_reversal(px, lookback=5)
sig_pairs = signals.cointegration_pairs_signal(px, max_pairs=50, lookback=252)


SIG = signals.combine_signals(
mom=(sig_mom, 0.5),
rev=(sig_rev, 0.3),
pairs=(sig_pairs, 0.2),
)


cfg = backtest.BacktestConfig(
rebal_days=REBAL_DAYS,
mu_half_life=63,
cov_lookback=COV_LOOKBACK,
risk_aversion=RISK_AVERSION,
l1_turnover=L1_TURNOVER,
cost_bps=COST_BPS,
lb=LB, ub=UB,
long_only=False,
net_exposure=NET,
turnover_cap=0.6,
liquidity_cap=(0.003 * adv.iloc[-1] / (adv.iloc[-1].sum() + 1e-12)) # simple proportional cap
)


rows = []
equities = {}
for name, cov_fn in risk.COV_FACTORIES.items():
print(f"Backtesting Σ = {name} ...")
res = backtest.run_backtest(px, SIG, cov_fn, cfg)
summ = evaluation.summarize_strategy(res["pnl"], name=name)
rows.append(summ)
equities[name] = res["cum"]


leader = pd.DataFrame(rows).set_index("name").sort_values("sharpe", ascending=False)
leader.to_csv("outputs/leaderboard.csv")
pd.DataFrame(equities).to_csv("outputs/equity_curves.csv")
print("\nLeaderboard (net):")
print(leader.round(3))
print("\nSaved: outputs/leaderboard.csv, outputs/equity_curves.csv")