from __future__ import annotations
cov_builder: Callable[[pd.DataFrame], pd.DataFrame],
cfg: BacktestConfig) -> Dict[str, pd.DataFrame | pd.Series | float]:
px = prices.copy().ffill()
ret = px.pct_change().fillna(0.0)
dates = ret.index


# standardize signals cross-sectionally and convert to expected returns by scaling
sig = signals.loc[dates].copy().fillna(0.0)
sig = sig.apply(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9), axis=1)
mu_daily = 0.05 * sig / 252.0


w = pd.DataFrame(0.0, index=dates, columns=ret.columns)
pnl = pd.Series(0.0, index=dates)
turnover = pd.Series(0.0, index=dates)
pred_vol = pd.Series(np.nan, index=dates)


w_prev = pd.Series(0.0, index=ret.columns)


for t in range(cfg.cov_lookback, len(dates)-1):
if (t - cfg.cov_lookback) % cfg.rebal_days != 0:
w.iloc[t] = w_prev
pnl.iloc[t+1] = float((w_prev * ret.iloc[t+1]).sum())
continue


window_ret = ret.iloc[t-cfg.cov_lookback+1:t+1]
mu_t = mu_daily.iloc[t]


S = cov_builder(window_ret)
pred_vol.iloc[t] = float(np.sqrt(np.maximum(1e-12, (w_prev.values @ S.values @ w_prev.values.T))))


sol = solve_qp(mu=mu_t,
Sigma=S,
w_prev=w_prev,
risk_aversion=cfg.risk_aversion,
l1_turnover=cfg.l1_turnover,
lb=cfg.lb, ub=cfg.ub,
long_only=cfg.long_only,
net_exposure=cfg.net_exposure,
turnover_cap=cfg.turnover_cap,
liquidity_cap=cfg.liquidity_cap)


dw = (sol - w_prev).abs().sum()
cost = (cfg.cost_bps * 1e-4) * dw
w_prev = sol
w.iloc[t] = sol
pnl.iloc[t+1] = float((sol * ret.iloc[t+1]).sum() - cost)
turnover.iloc[t] = float(dw)


cum = (1 + pnl.fillna(0.0)).cumprod()
out = {
"weights": w,
"pnl": pnl,
"cum": cum,
"turnover": turnover,
"pred_vol": pred_vol,
}
return out