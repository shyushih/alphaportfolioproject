from __future__ import annotations
turnover_cap: float | None = 0.5,
liquidity_cap: pd.Series | None = None,
) -> pd.Series:
"""
Maximize mu'w - lambda w'Sigma w - gamma * L1_turnover
subject to bounds, sum(w)=net_exposure, and optional caps.
"""
idx = mu.index
N = len(idx)
S = Sigma.loc[idx, idx].to_numpy()
# ensure PD-ish
eigmin = np.linalg.eigvalsh(S).min()
if eigmin < 1e-8:
S = S + (1e-6 - eigmin) * np.eye(N)


w = cp.Variable(N)
obj = mu.values @ w - risk_aversion * cp.quad_form(w, S)


if w_prev is None:
w_prev = pd.Series(0.0, index=idx)
dw = w - w_prev.values


if l1_turnover > 0:
obj -= l1_turnover * cp.norm1(dw)


constraints = []
if long_only:
constraints += [w >= max(0.0, lb)]
constraints += [w <= max(0.0, ub)]
else:
constraints += [w >= lb, w <= ub]


if net_exposure is not None:
constraints += [cp.sum(w) == net_exposure]


if turnover_cap is not None:
constraints += [cp.norm1(dw) <= turnover_cap]


if liquidity_cap is not None:
caps = liquidity_cap.loc[idx].fillna(abs(ub)).values
constraints += [cp.abs(w) <= caps]


prob = cp.Problem(cp.Maximize(obj), constraints)
try:
prob.solve(solver=cp.OSQP, verbose=False)
except Exception:
pass
if w.value is None:
try:
prob.solve(solver=cp.ECOS, verbose=False)
except Exception:
pass
if w.value is None:
raise RuntimeError("QP failed to solve")


sol = pd.Series(w.value, index=idx)
return sol