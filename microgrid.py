"""Prototype for the TB-Science proposal: risk-averse multistage storage dispatch
under regime-switching price spikes.

Contains the synthetic generator, the battery/grid simulator, five dispatch
policies and the replay harness. Everything is numpy/scipy only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

HOURS_PER_DAY = 24


# ----------------------------------------------------------------------------
# System specification
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    e_max: float = 20.0          # MWh usable capacity
    soc_min: float = 2.0         # MWh
    soc_max: float = 18.0        # MWh
    p_max: float = 5.0           # MW charge / discharge limit
    imp_max: float = 8.0         # MW grid import limit
    exp_max: float = 6.0         # MW grid export limit
    deg_cost: float = 8.0        # $/MWh throughput (cycling degradation)
    export_factor: float = 0.95  # export price = factor * spot
    soc_init: float = 10.0       # MWh at start of each evaluation path

    def eta_charge(self, p: float, soc: float) -> float:
        """Charge efficiency: falls with power and near the top of the SOC window."""
        frac = soc / self.e_max
        return 0.96 - 0.10 * (p / self.p_max) - 0.05 * max(0.0, frac - 0.8) / 0.2

    def eta_discharge(self, p: float, soc: float) -> float:
        frac = soc / self.e_max
        return 0.96 - 0.10 * (p / self.p_max) - 0.05 * max(0.0, 0.2 - frac) / 0.2


SPEC = Spec()


# ----------------------------------------------------------------------------
# Synthetic generator (hidden from the agent in the real task)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GenParams:
    p_normal_to_spike: float = 0.010     # per hour
    p_spike_to_normal: float = 0.10      # per hour (mean spike ~10 h)
    base_price: float = 45.0
    annual_amp: float = 8.0
    diurnal_amp: float = 18.0
    netload_beta: float = 6.0            # $/MWh per MW of net load above mean
    ar_price: float = 0.75
    sd_price: float = 5.0
    spike_mult: float = 1.6
    spike_jump_mean: float = 120.0
    sd_spike: float = 25.0
    wind_cap: float = 10.0
    load_mean: float = 6.0
    # forecast error structure
    fc_price_bias: float = 1.0
    fc_price_sd_normal: float = 6.0
    fc_price_sd_spike: float = 6.0
    fc_spike_capture: float = 0.0        # day-ahead forecast does not see spike timing; only realised history reveals the regime
    fc_wind_sd: float = 1.2
    fc_load_sd: float = 0.5


GEN = GenParams()


def _diurnal(hour: np.ndarray) -> np.ndarray:
    # evening peak around 18h, trough around 4h
    return np.sin((hour - 10) / 24 * 2 * np.pi) * 0.6 + np.sin((hour - 16) / 12 * 2 * np.pi) * 0.4


def generate(n_hours: int, seed: int, start_hour: int = 0, regime0: int = 0,
             gp: GenParams = GEN) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    abs_hour = start_hour + t
    hod = abs_hour % 24
    doy = (abs_hour // 24) % 365

    # latent regime chain
    regime = np.zeros(n_hours, dtype=int)
    r = regime0
    for i in range(n_hours):
        if r == 0 and rng.random() < gp.p_normal_to_spike:
            r = 1
        elif r == 1 and rng.random() < gp.p_spike_to_normal:
            r = 0
        regime[i] = r

    # wind: seasonal (higher in winter) + AR(1) in logit-ish space
    wind_season = 0.45 + 0.15 * np.cos(doy / 365 * 2 * np.pi)
    w = np.zeros(n_hours)
    x = 0.0
    for i in range(n_hours):
        x = 0.85 * x + rng.normal(0, 0.35)
        w[i] = gp.wind_cap * np.clip(wind_season[i] + 0.25 * np.tanh(x) , 0.02, 0.98)

    # load: diurnal profile + weekly + AR noise
    weekday = ((abs_hour // 24) % 7) < 5
    load = gp.load_mean * (1 + 0.25 * _diurnal(hod) + 0.08 * np.cos(doy / 365 * 2 * np.pi)) \
        * np.where(weekday, 1.0, 0.92)
    e = 0.0
    for i in range(n_hours):
        e = 0.7 * e + rng.normal(0, 0.25)
        load[i] = max(1.0, load[i] + e)

    netload = load - w
    season_price = gp.base_price + gp.annual_amp * np.cos(doy / 365 * 2 * np.pi) \
        + gp.diurnal_amp * _diurnal(hod) + gp.netload_beta * (netload - netload.mean())

    price = np.zeros(n_hours)
    resid = 0.0
    jump = 0.0
    for i in range(n_hours):
        if regime[i] == 0:
            resid = gp.ar_price * resid + rng.normal(0, gp.sd_price)
            price[i] = season_price[i] + resid
            jump = 0.0
        else:
            if jump == 0.0:
                jump = rng.exponential(gp.spike_jump_mean)
            else:
                jump = 0.85 * jump + rng.exponential(gp.spike_jump_mean) * 0.15
            price[i] = season_price[i] * gp.spike_mult + jump + rng.normal(0, gp.sd_spike)
    price = np.maximum(price, -20.0)

    # day-ahead forecasts: structured errors; spikes only partly foreseen
    fc_price = season_price + gp.fc_price_bias + np.where(
        regime == 1,
        gp.fc_spike_capture * (price - season_price) + rng.normal(0, gp.fc_price_sd_spike, n_hours),
        rng.normal(0, gp.fc_price_sd_normal, n_hours))
    fc_wind = np.clip(w + _ar_noise(rng, n_hours, 0.6, gp.fc_wind_sd), 0, gp.wind_cap)
    fc_load = np.maximum(1.0, load + _ar_noise(rng, n_hours, 0.6, gp.fc_load_sd))

    return pd.DataFrame({
        "hour": abs_hour, "price": price, "wind": w, "load": load,
        "fc_price": fc_price, "fc_wind": fc_wind, "fc_load": fc_load,
        "regime": regime,   # dropped before the agent sees the data
    })


def _ar_noise(rng, n, phi, sd):
    out = np.zeros(n)
    x = 0.0
    for i in range(n):
        x = phi * x + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
        out[i] = x
    return out


# ----------------------------------------------------------------------------
# Simulator / replay harness
# ----------------------------------------------------------------------------
def step(spec: Spec, soc: float, charge: float, discharge: float,
         price: float, wind: float, load: float):
    """Apply one hour. Returns (new_soc, cost, violations)."""
    viol = 0
    charge = float(np.clip(charge, 0, spec.p_max))
    discharge = float(np.clip(discharge, 0, spec.p_max))
    if charge > 0 and discharge > 0:   # no simultaneous charge/discharge
        if charge >= discharge:
            charge, discharge = charge - discharge, 0.0
        else:
            charge, discharge = 0.0, discharge - charge
    eta_c = spec.eta_charge(charge, soc)
    eta_d = spec.eta_discharge(discharge, soc)
    new_soc = soc + eta_c * charge - discharge / eta_d
    if new_soc > spec.soc_max + 1e-6:
        viol += 1
        charge = max(0.0, (spec.soc_max - soc) / eta_c)
        new_soc = spec.soc_max
    if new_soc < spec.soc_min - 1e-6:
        viol += 1
        discharge = max(0.0, (soc - spec.soc_min) * eta_d)
        new_soc = spec.soc_min
    net = load - wind + charge - discharge   # MW from grid (positive = import)
    if net > spec.imp_max + 1e-6:
        viol += 1
        net = spec.imp_max                    # curtail charging (real system would trip)
    if net < -spec.exp_max - 1e-6:
        viol += 1
        net = -spec.exp_max
    energy_cost = price * net if net >= 0 else spec.export_factor * price * net
    cost = energy_cost + spec.deg_cost * (charge + discharge)
    return new_soc, cost, viol


def replay(policy, path: pd.DataFrame, spec: Spec = SPEC, horizon: int = 24):
    """Roll a policy through one path hour by hour. Policy sees only the past and forecasts."""
    n = len(path) - horizon
    soc = spec.soc_init
    costs = np.zeros(n)
    viols = 0
    for t in range(n):
        state = {"soc": soc, "t": t}
        history = path.iloc[:t][["price", "wind", "load", "fc_price", "fc_wind", "fc_load"]]
        forecasts = path.iloc[t:t + horizon][["fc_price", "fc_wind", "fc_load"]].to_numpy()
        charge, discharge = policy.decide(state, history, forecasts)
        soc, c, v = step(spec, soc, charge, discharge,
                         path.price.iat[t], path.wind.iat[t], path.load.iat[t])
        costs[t] = c
        viols += v
    daily = costs[: (n // HOURS_PER_DAY) * HOURS_PER_DAY].reshape(-1, HOURS_PER_DAY).sum(1)
    return daily, viols


def cvar(x: np.ndarray, alpha: float = 0.95) -> float:
    x = np.sort(x)
    k = max(1, int(np.ceil((1 - alpha) * len(x))))
    return float(x[-k:].mean())


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
class DoNothing:
    name = "do-nothing"

    def calibrate(self, train): pass

    def decide(self, state, history, forecasts):
        return 0.0, 0.0


class GreedyThreshold:
    name = "greedy threshold"

    def calibrate(self, train):
        self.lo = np.quantile(train.fc_price, 0.30)
        self.hi = np.quantile(train.fc_price, 0.70)

    def decide(self, state, history, forecasts):
        p = forecasts[0, 0]
        if p < self.lo:
            return SPEC.p_max, 0.0
        if p > self.hi:
            return 0.0, SPEC.p_max
        return 0.0, 0.0


class _LPBase:
    """Shared LP builder: K scenarios, horizon H, first-stage (c0, d0) shared."""
    eta_nom = 0.92
    piecewise = False
    lam = 0.0
    alpha = 0.95
    soc_margin = 0.0      # safety margin inside the SOC window (expert practice)
    terminal_frac = 0.8   # value of stored energy at horizon end, as a fraction of mean scenario price

    def _build(self, soc0, price_s, wind_s, load_s, spec: Spec):
        K, H = price_s.shape
        pw = 2 if self.piecewise else 1
        # per scenario vars: c[pw*H], d[pw*H], imp[H], exp[H], soc[H]
        nv_s = (2 * pw + 3) * H
        n = K * nv_s + (K + 1 if self.lam > 0 else 0)
        c = np.zeros(n)
        lb = np.zeros(n)
        ub = np.full(n, np.inf)

        def idx(s, kind, h, seg=0):
            base = s * nv_s
            if kind == "c":
                return base + seg * H + h
            if kind == "d":
                return base + pw * H + seg * H + h
            if kind == "imp":
                return base + 2 * pw * H + h
            if kind == "exp":
                return base + 2 * pw * H + H + h
            if kind == "soc":
                return base + 2 * pw * H + 2 * H + h
            raise KeyError(kind)

        if self.piecewise:
            segs = [(0.95, spec.p_max / 2), (0.86, spec.p_max / 2)]
        else:
            segs = [(self.eta_nom, spec.p_max)]

        n_eq = K * (2 * H)
        A = lil_matrix((n_eq, n))
        b = np.zeros(n_eq)
        row = 0
        wgt = 1.0 / K
        for s in range(K):
            for h in range(H):
                for seg, (eta, cap) in enumerate(segs):
                    ub[idx(s, "c", h, seg)] = cap
                    ub[idx(s, "d", h, seg)] = cap
                    c[idx(s, "c", h, seg)] += wgt * spec.deg_cost
                    c[idx(s, "d", h, seg)] += wgt * spec.deg_cost
                ub[idx(s, "imp", h)] = spec.imp_max
                ub[idx(s, "exp", h)] = spec.exp_max
                lb[idx(s, "soc", h)] = spec.soc_min + self.soc_margin
                ub[idx(s, "soc", h)] = spec.soc_max - self.soc_margin
                c[idx(s, "imp", h)] += wgt * price_s[s, h]
                c[idx(s, "exp", h)] -= wgt * spec.export_factor * price_s[s, h]
                if h == H - 1:   # terminal value of stored energy
                    c[idx(s, "soc", h)] -= wgt * self.terminal_frac * price_s[s].mean()
                # balance: imp - exp - sum c + sum d = load - wind
                A[row, idx(s, "imp", h)] = 1
                A[row, idx(s, "exp", h)] = -1
                for seg in range(len(segs)):
                    A[row, idx(s, "c", h, seg)] = -1
                    A[row, idx(s, "d", h, seg)] = 1
                b[row] = load_s[s, h] - wind_s[s, h]
                row += 1
                # soc dynamics
                A[row, idx(s, "soc", h)] = 1
                if h > 0:
                    A[row, idx(s, "soc", h - 1)] = -1
                for seg, (eta, cap) in enumerate(segs):
                    A[row, idx(s, "c", h, seg)] = -eta
                    A[row, idx(s, "d", h, seg)] = 1 / eta
                b[row] = soc0 if h == 0 else 0.0
                row += 1

        # non-anticipativity on first-stage charge/discharge
        na_rows = []
        for s in range(1, K):
            for seg in range(len(segs)):
                for kind in ("c", "d"):
                    r = np.zeros(n)
                    r[idx(0, kind, 0, seg)] = 1
                    r[idx(s, kind, 0, seg)] = -1
                    na_rows.append(r)
        A_ub, b_ub = None, None
        if self.lam > 0:
            # CVaR: z at n-K-1, u_s at n-K+s ; cost_s - z - u_s <= 0
            z = n - K - 1
            lb[z] = -np.inf
            c[z] += self.lam
            rows = []
            for s in range(K):
                u = n - K + s
                c[u] += self.lam / ((1 - self.alpha) * K)
                r = np.zeros(n)
                for h in range(H):
                    r[idx(s, "imp", h)] = price_s[s, h]
                    r[idx(s, "exp", h)] = -spec.export_factor * price_s[s, h]
                    for seg in range(len(segs)):
                        r[idx(s, "c", h, seg)] = spec.deg_cost
                        r[idx(s, "d", h, seg)] = spec.deg_cost
                r[z] = -1
                r[u] = -1
                rows.append(r)
            A_ub = np.array(rows)
            b_ub = np.zeros(K)
        A_eq = csr_matrix(A)
        if na_rows:
            A_eq = csr_matrix(np.vstack([A.toarray(), np.array(na_rows)]))
            b = np.concatenate([b, np.zeros(len(na_rows))])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b,
                      bounds=list(zip(lb, ub)), method="highs")
        if not res.success:
            return 0.0, 0.0
        x = res.x
        ch = sum(x[idx(0, "c", 0, seg)] for seg in range(len(segs)))
        di = sum(x[idx(0, "d", 0, seg)] for seg in range(len(segs)))
        return float(ch), float(di)


class DeterministicMPC(_LPBase):
    """Point-forecast MPC with nominal constant efficiency. The first thing one builds."""
    name = "point-forecast MPC"

    def calibrate(self, train): pass

    def decide(self, state, history, forecasts):
        f = forecasts
        return self._build(state["soc"], f[:, 0][None, :], f[:, 1][None, :], f[:, 2][None, :], SPEC)


class ScenarioModel:
    """Regime-aware forecast-error model learned from training data only."""

    def fit(self, train: pd.DataFrame, rng):
        self.rng = rng
        err = (train.price - train.fc_price).to_numpy()
        med = np.median(err)
        mad = np.median(np.abs(err - med)) * 1.4826
        spike = err > med + 3.0 * mad
        # transition probabilities from the classified sequence
        s = spike.astype(int)
        n01 = np.sum((s[:-1] == 0) & (s[1:] == 1)); n0 = np.sum(s[:-1] == 0)
        n10 = np.sum((s[:-1] == 1) & (s[1:] == 0)); n1 = np.sum(s[:-1] == 1)
        self.p01 = (n01 + 1) / (n0 + 2)
        self.p10 = (n10 + 1) / (n1 + 2)
        self.err_normal = err[~spike]
        self.err_spike = err[spike] if spike.sum() > 20 else err[err > np.quantile(err, 0.97)]
        self.mu0, self.sd0 = self.err_normal.mean(), self.err_normal.std()
        self.mu1, self.sd1 = self.err_spike.mean(), max(self.err_spike.std(), 1.0)
        # AR(1) persistence of normal errors
        e = self.err_normal
        self.phi = float(np.corrcoef(e[:-1], e[1:])[0, 1])
        werr = (train.wind - train.fc_wind).to_numpy()
        lerr = (train.load - train.fc_load).to_numpy()
        self.werr, self.lerr = werr, lerr
        self.stationary = self.p01 / (self.p01 + self.p10)

    def filter_regime(self, history: pd.DataFrame, window: int = 48) -> float:
        """Posterior P(spike now) via a forward filter on recent forecast errors."""
        if len(history) == 0:
            return self.stationary
        err = (history.price - history.fc_price).to_numpy()[-window:]
        p = self.stationary
        for e in err:
            prior1 = p * (1 - self.p10) + (1 - p) * self.p01
            l0 = np.exp(-0.5 * ((e - self.mu0) / self.sd0) ** 2) / self.sd0
            l1 = np.exp(-0.5 * ((e - self.mu1) / self.sd1) ** 2) / self.sd1
            p = prior1 * l1 / (prior1 * l1 + (1 - prior1) * l0 + 1e-300)
        return float(p)

    def sample(self, forecasts: np.ndarray, K: int, p_spike0: float):
        H = forecasts.shape[0]
        price = np.empty((K, H)); wind = np.empty((K, H)); load = np.empty((K, H))
        for k in range(K):
            r = int(self.rng.random() < p_spike0)
            e = self.rng.choice(self.err_normal)
            for h in range(H):
                if h > 0:
                    if r == 0 and self.rng.random() < self.p01:
                        r = 1
                    elif r == 1 and self.rng.random() < self.p10:
                        r = 0
                if r == 0:
                    e = self.phi * e + np.sqrt(1 - self.phi ** 2) * self.rng.choice(self.err_normal)
                    price[k, h] = forecasts[h, 0] + e
                else:
                    price[k, h] = forecasts[h, 0] + self.rng.choice(self.err_spike)
            wind[k] = np.clip(forecasts[:, 1] + self.rng.choice(self.werr, H), 0, None)
            load[k] = np.maximum(1.0, forecasts[:, 2] + self.rng.choice(self.lerr, H))
        return price, wind, load


class TwoStageRiskNeutral(_LPBase):
    """Scenarios drawn from the unconditional error distribution, risk-neutral, nominal efficiency."""
    name = "two-stage, risk-neutral"
    K = 20

    def calibrate(self, train):
        self.model = ScenarioModel(); self.model.fit(train, np.random.default_rng(7))

    def decide(self, state, history, forecasts):
        p, w, l = self.model.sample(forecasts, self.K, self.model.stationary)
        return self._build(state["soc"], p, w, l, SPEC)


class RegimeAwareCVaR(_LPBase):
    """Reference: regime-filtered scenarios, CVaR-penalised objective, piecewise efficiency."""
    name = "regime-aware CVaR (reference)"
    K = 20
    lam = 1.0
    piecewise = True
    soc_margin = 0.4

    def calibrate(self, train):
        self.model = ScenarioModel(); self.model.fit(train, np.random.default_rng(11))

    def decide(self, state, history, forecasts):
        p_spike = self.model.filter_regime(history)
        p, w, l = self.model.sample(forecasts, self.K, p_spike)
        return self._build(state["soc"], p, w, l, SPEC)


POLICIES = [DoNothing, GreedyThreshold, DeterministicMPC, TwoStageRiskNeutral, RegimeAwareCVaR]
