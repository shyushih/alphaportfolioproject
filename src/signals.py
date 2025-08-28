from __future__ import annotations
r_1 = prices.pct_change(21)
sig = r_12 - r_1
return sig


def short_term_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
return -prices.pct_change(lookback)


def volatility_inverse(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
vol = prices.pct_change().rolling(window).std()
return 1.0 / (vol.replace(0, np.nan))


def zscore(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
mu = df.rolling(window).mean()
sd = df.rolling(window).std()
return (df - mu) / sd


def cointegration_pairs_signal(prices: pd.DataFrame, max_pairs: int = 50, lookback: int = 252):
"""
Toy Engle–Granger pair signal:
1) pick top correlated pairs over lookback
2) filter by coint p-value < 0.05
3) signal = -zscore(spread) (mean reversion)
Returns a DataFrame of signals per asset (sum of pair signals touching each asset).
"""
px = prices.dropna(axis=1, how="any").iloc[-lookback:]
if px.shape[1] < 3:
return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
ret = px.pct_change().dropna()
corr = ret.corr().abs()
np.fill_diagonal(corr.values, 0.0)
pairs = (
corr.stack()
.sort_values(ascending=False)
.index
.unique()[:max_pairs]
)
sig = pd.DataFrame(0.0, index=px.index, columns=px.columns, dtype=float)
for a,b in pairs:
s1, s2 = px[a], px[b]
pv = coint(s1, s2, trend="c")[1]
if pv < 0.05:
# hedge ratio via OLS beta
x = np.vstack([s2.values, np.ones(len(s2))]).T
beta, alpha = np.linalg.lstsq(x, s1.values, rcond=None)[0]
spread = s1 - (alpha + beta*s2)
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
sig[a] = sig[a].add(-z, fill_value=0.0)
sig[b] = sig[b].add(z, fill_value=0.0)
# align to full index
out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
out.loc[sig.index, sig.columns] = sig
return out.ffill().fillna(0.0)


def combine_signals(**kwargs) -> pd.DataFrame:
"""
Weighted linear combo of provided signals (DataFrames aligned on index/columns).
Example: combine_signals(mom=0.5, rev=0.3, pairs=0.2)
kwargs should be name=(df, weight)
"""
acc = None
for name, tup in kwargs.items():
df, w = tup
df = df.copy()
if acc is None:
acc = w * df
else:
acc = acc.add(w*df, fill_value=0.0)
return acc