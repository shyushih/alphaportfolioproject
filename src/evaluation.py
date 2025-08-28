from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import annualize_ret_vol, max_drawdown


def summarize_strategy(pnl: pd.Series, name: str = "strategy") -> pd.Series:
r = pnl.dropna()
mu, vol = annualize_ret_vol(r)
dd = max_drawdown((1+r).cumprod())
sr = mu / (vol + 1e-12)
calmar = -mu / (dd if dd != 0 else np.nan)
return pd.Series({
"name": name,
"ann_return": mu,
"ann_vol": vol,
"sharpe": sr,
"max_drawdown": dd,
"calmar": calmar,
})


def realized_vs_pred_vol(pnl: pd.Series, pred_vol: pd.Series, window: int = 63):
real_vol = pnl.rolling(window).std()
return pd.DataFrame({
"realized_vol": real_vol,
"pred_vol": pred_vol
}).dropna()