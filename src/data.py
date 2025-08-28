from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf


def download_prices(tickers, start: str, end: str) -> pd.DataFrame:
"""
Returns Close prices for `tickers` between start and end, business days.
"""
df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
if isinstance(df, pd.Series):
df = df.to_frame()
df = df.sort_index().asfreq("B").ffill()
return df


def download_volume(tickers, start: str, end: str) -> pd.DataFrame:
vol = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Volume"]
if isinstance(vol, pd.Series):
vol = vol.to_frame()
vol = vol.sort_index().asfreq("B").ffill()
return vol


def compute_adv(prices: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
"""
Approximate ADV in dollars: rolling mean of price * volume.
"""
prices, volume = prices.align(volume, join="inner", axis=0)
adv = (prices * volume).rolling(window).mean()
return adv


def basic_universe_filter(prices: pd.DataFrame, adv: pd.DataFrame, min_price=3.0, min_adv=5e6):
last_px = prices.ffill().iloc[-1]
last_adv = adv.ffill().iloc[-1]
keep = (last_px > min_price) & (last_adv > min_adv)
return sorted(list(last_px[keep].index))