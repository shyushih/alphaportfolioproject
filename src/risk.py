from __future__ import annotations
x = r[t:t+1].T
S = lam * S + (1-lam) * (x @ x.T)
S = S * (1-lam) / (1 - lam**T + 1e-12)
return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
lw = LedoitWolf().fit(returns.fillna(0.0).to_numpy())
S = lw.covariance_
return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def oas_cov(returns: pd.DataFrame) -> pd.DataFrame:
o = OAS().fit(returns.fillna(0.0).to_numpy())
S = o.covariance_
return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def pca_denoise_cov(returns: pd.DataFrame, n_factors: int = 5, ridge: float = 1e-6) -> pd.DataFrame:
R = returns.fillna(0.0)
X = R.to_numpy()
pca = PCA(n_components=min(n_factors, X.shape[1]-1))
F = pca.fit_transform(X)
B = np.linalg.lstsq(F, X, rcond=None)[0]
X_hat = F @ B
resid = X - X_hat
S_f = np.cov(X_hat, rowvar=False)
S_e = np.diag(resid.var(axis=0))
S = S_f + S_e + ridge*np.eye(X.shape[1])
return pd.DataFrame(S, index=R.columns, columns=R.columns)


def graphical_lasso_cov(returns: pd.DataFrame) -> pd.DataFrame:
X = returns.fillna(0.0).to_numpy()
gl = GraphicalLassoCV().fit(X)
S = gl.covariance_
return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def mp_denoise_cov(returns: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
R = returns.fillna(0.0)
X = R.to_numpy()
T, N = X.shape
c = N / max(T, 1)
S = np.cov(X, rowvar=False)
vals, vecs = np.linalg.eigh(S)
sigma2 = np.median(vals) # crude noise level proxy
lambda_plus = sigma2 * (1 + np.sqrt(c))**2
bulk = vals <= lambda_plus
if bulk.any():
vals[bulk] = vals[bulk].mean()
S_shrunk = (vecs @ np.diag(vals) @ vecs.T) + ridge*np.eye(N)
return pd.DataFrame(S_shrunk, index=R.columns, columns=R.columns)


COV_FACTORIES = {
"sample": sample_cov,
"ewma": ewma_cov,
"ledoit_wolf": ledoit_wolf_cov,
"oas": oas_cov,
"pca_denoise": pca_denoise_cov,
"graph_lasso": graphical_lasso_cov,
"mp_denoise": mp_denoise_cov,
}