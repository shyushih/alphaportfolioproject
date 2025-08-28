import numpy as np
import pandas as pd


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
return prices.pct_change().dropna(how="all")


def annualize_ret_vol(daily_ret: pd.Series, trading_days: int = 252):
mu = daily_ret.mean() * trading_days
vol = daily_ret.std(ddof=0) * np.sqrt(trading_days)
return mu, vol


def max_drawdown(cum: pd.Series) -> float:
roll_max = cum.cummax()
dd = (cum / roll_max) - 1.0
return dd.min()


def winsorize(s: pd.Series, p=0.01):
lo, hi = s.quantile([p, 1-p])
return s.clip(lo, hi)


def align_index(*dfs):
idx = dfs[0].index
for d in dfs[1:]:
idx = idx.intersection(d.index)
return [d.loc[idx] for d in dfs]


def safe_fillna(df: pd.DataFrame, method="ffill"):
if method == "ffill":
return df.ffill()
if method == "zero":
return df.fillna(0.0)
return df