# -*- coding: utf-8 -*-
"""Phase 6B.1 — Dashboard proxy CSV generator (v2 — realistic oscillation).

Generates dashboard/data/ess_dashboard_data.csv: 5 ESS systems × 365 days
= 1,825 rows. Each system has a distinct narrative character so the
demo dashboard can show different status colour-codes side-by-side
during the GAIP 2026 walkthrough.

v2 changes (realistic oscillation):
  - Precursor trajectories use Ornstein-Uhlenbeck mean-reverting walks
    instead of monotonic linear growth → daily up-AND-down movement
  - Discrete maintenance events drop cell-variance / SOC envelope /
    thermal at scheduled dates → premium DIPS visible across the year
  - Behavioural feedback: when s_overcharge stays elevated for 7+ days
    the operator tightens SOC envelope (s_overcharge drops 60%) → models
    the customer-experience loop the paper claims
  - Weekly operational cycle (small sin-wave) added to all precursors
    → realistic short-period jitter on top of long trend

System characters:
  ESS-01 — "Steady Performer"        : OU around baseline, small wobbles
  ESS-02 — "Aging Fast w/ Maintenance": upward drift + 3 maintenance dips
  ESS-03 — "SOC Volatility"          : episodic SOC spikes that resolve
  ESS-04 — "Summer Thermal Stress"   : summer ramp + autumn recovery
  ESS-05 — "Cell Imbalance Drift"    : quarterly maintenance sawtooth

Premium derivation uses the same Tower A posterior parameters as the
paper (β₀^fire = -12.8957, β₁ = 0.4291). For demo readability the
output is mapped onto a $400-$2,000 range (raw model output is $510-520
which is invisible at presentation distance). The unamplified Tower A
P(fire) is preserved per-row for transparency.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent.parent / "data" / "ess_dashboard_data.csv"

# Tower A posterior point estimates
BETA0_FIRE = -12.8957
BETA1 = 0.4291

# Insurance-pricing parameters (paper §3.4)
LGD_FIRE = 0.60
RISK_LOADING = 0.30
ADMIN_BPS_PER_WEEK = 0.0001
HRR_REFERENCE_KJ = 1200

TIV_USD = 5_000_000

# Date range — demo year, ending 2026-04-30 (today is 2026-05-13)
DATE_END = pd.Timestamp("2026-04-30")
DATE_START = DATE_END - pd.Timedelta(days=364)
N_DAYS = 365

# Status thresholds on aggregate s_x score
NORMAL_MAX = 0.5
WARNING_MAX = 1.5

# 5 system narrative characters
SYSTEM_PROFILES = [
    {
        "system_id": 1,
        "system_name": "ESS-01 Steady Performer",
        "soh_start": 0.99, "soh_end": 0.985,
        "char": "steady",
    },
    {
        "system_id": 2,
        "system_name": "ESS-02 Aging Fast",
        "soh_start": 0.99, "soh_end": 0.94,
        "char": "aging_with_maintenance",
    },
    {
        "system_id": 3,
        "system_name": "ESS-03 SOC Volatility",
        "soh_start": 0.985, "soh_end": 0.97,
        "char": "soc_episodic",
    },
    {
        "system_id": 4,
        "system_name": "ESS-04 Summer Thermal Stress",
        "soh_start": 0.985, "soh_end": 0.965,
        "char": "summer_thermal",
    },
    {
        "system_id": 5,
        "system_name": "ESS-05 Cell Imbalance Drift",
        "soh_start": 0.99, "soh_end": 0.975,
        "char": "imbalance_sawtooth",
    },
]


def ou_process(rng, n, mu, theta=0.10, sigma=0.10, x0=None):
    """Ornstein-Uhlenbeck mean-reverting walk.
       x_{t+1} = x_t + theta*(mu_t - x_t) + sigma * N(0,1)
       mu can be a scalar or length-n array (time-varying mean)."""
    mu_arr = np.full(n, mu) if np.isscalar(mu) else np.asarray(mu)
    x = np.zeros(n)
    x[0] = x0 if x0 is not None else mu_arr[0]
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu_arr[t] - x[t-1]) + sigma * rng.standard_normal()
    return x


def seasonal_temp_array(boost_c, n=N_DAYS, base_c=24.0):
    """Sin-wave seasonal temperature for the year.
       Day 0 = May 1 (mid-spring); peak summer at day ~100 (mid-Aug)."""
    days = np.arange(n)
    phase = (days - 100) / 365.0 * 2 * np.pi
    return base_c + boost_c * np.cos(phase)


def weekly_cycle(n=N_DAYS, amp=0.08):
    """Small weekly operational cycle (7-day sin)."""
    days = np.arange(n)
    return amp * np.sin(2 * np.pi * days / 7.0)


def apply_maintenance_event(arr, day, drop_factor=0.4, recovery_days=21):
    """Drop value at `day` to `drop_factor * arr[day]` then exponentially
    recover toward original trajectory over `recovery_days`."""
    if day >= len(arr):
        return arr
    out = arr.copy()
    drop_amount = arr[day] * (1 - drop_factor)  # how much to subtract
    for i in range(day, min(len(arr), day + recovery_days * 2)):
        days_since = i - day
        # exponential decay of the maintenance benefit
        residual = drop_amount * np.exp(-days_since / (recovery_days / 2.0))
        out[i] = arr[i] - residual
    return out


def trajectory_for_character(char, rng):
    """Generate (cell_var, soc_max_drift, soc_volatility, thermal_anomaly,
    isc_anomaly) trajectories of length N_DAYS for the system character.
    All quantities are arrays — premium is derived per-day downstream."""
    n = N_DAYS

    if char == "steady":
        # Everything hovers at baseline with small mean-reverting noise
        cell_var = ou_process(rng, n, mu=1.0, theta=0.15, sigma=0.06, x0=1.0)
        soc_max_drift = ou_process(rng, n, mu=0.85, theta=0.20, sigma=0.015, x0=0.85)
        soc_vol = ou_process(rng, n, mu=0.05, theta=0.20, sigma=0.012, x0=0.05)
        thermal_anom = ou_process(rng, n, mu=0.0, theta=0.20, sigma=0.10, x0=0.0)
        isc_anom = ou_process(rng, n, mu=0.0, theta=0.20, sigma=0.08, x0=0.0)
        # No maintenance events needed — already steady

    elif char == "aging_with_maintenance":
        # Cell variance climbs from 1.0 → 3.5 over year, BUT with 3
        # maintenance events that drop variance by 50% temporarily
        cell_var_trend = np.linspace(1.0, 3.5, n)
        cell_var = ou_process(rng, n, mu=cell_var_trend, theta=0.08, sigma=0.18,
                              x0=1.0)
        # 3 maintenance events at days 80, 180, 280
        for day in (80, 180, 280):
            cell_var = apply_maintenance_event(cell_var, day,
                                               drop_factor=0.55, recovery_days=35)
        soc_max_drift = ou_process(rng, n, mu=0.88, theta=0.15, sigma=0.04)
        soc_vol = ou_process(rng, n, mu=0.10, theta=0.18, sigma=0.025)
        thermal_anom = ou_process(rng, n, mu=0.3, theta=0.15, sigma=0.20)
        isc_anom = (cell_var - 1.0) * 0.3 + ou_process(rng, n, mu=0.0,
                                                       theta=0.20, sigma=0.20)

    elif char == "soc_episodic":
        # Episodic SOC spikes — 6 random episodes per year, each lasting
        # 8-15 days, where soc_max climbs to 0.97 then resolves
        soc_max_drift = ou_process(rng, n, mu=0.86, theta=0.20, sigma=0.025)
        ep_days = rng.choice(range(20, n - 20), size=6, replace=False)
        for ep in ep_days:
            length = rng.integers(8, 16)
            for i in range(length):
                if ep + i < n:
                    soc_max_drift[ep + i] += 0.10 * np.exp(-i / (length / 2.0))
        cell_var = ou_process(rng, n, mu=1.3, theta=0.15, sigma=0.18)
        soc_vol = ou_process(rng, n, mu=0.18, theta=0.18, sigma=0.05)
        thermal_anom = ou_process(rng, n, mu=0.2, theta=0.18, sigma=0.20)
        isc_anom = ou_process(rng, n, mu=0.0, theta=0.20, sigma=0.15)

    elif char == "summer_thermal":
        # Strong seasonal thermal trajectory: low winter, ramp Jun-Aug,
        # peak Aug, recovery Sep-Nov via cooling-system maintenance event
        thermal_anom = ou_process(rng, n,
                                  mu=2.4 * np.maximum(0, np.sin(
                                      np.linspace(np.pi * 0.10, np.pi * 1.55, n)
                                  )),
                                  theta=0.10, sigma=0.30)
        # Cooling-system upgrade at day 140 (mid-Sep) drops thermal anomaly
        thermal_anom = apply_maintenance_event(thermal_anom, 140,
                                               drop_factor=0.35, recovery_days=25)
        cell_var = ou_process(rng, n, mu=1.5 + 0.4 * (np.arange(n) / n),
                              theta=0.12, sigma=0.18)
        soc_max_drift = ou_process(rng, n, mu=0.86, theta=0.20, sigma=0.03)
        soc_vol = ou_process(rng, n, mu=0.08, theta=0.20, sigma=0.020)
        isc_anom = ou_process(rng, n, mu=0.0, theta=0.20, sigma=0.15)

    elif char == "imbalance_sawtooth":
        # Cell imbalance climbs steadily but quarterly maintenance events
        # drop it 70% → sawtooth pattern
        cell_var_trend = np.linspace(1.2, 4.5, n)
        cell_var = ou_process(rng, n, mu=cell_var_trend, theta=0.08, sigma=0.20,
                              x0=1.2)
        # Quarterly events: days 90, 180, 270
        for day in (90, 180, 270):
            cell_var = apply_maintenance_event(cell_var, day,
                                               drop_factor=0.40, recovery_days=45)
        soc_max_drift = ou_process(rng, n, mu=0.86, theta=0.18, sigma=0.025)
        soc_vol = ou_process(rng, n, mu=0.07, theta=0.20, sigma=0.018)
        thermal_anom = ou_process(rng, n, mu=0.3, theta=0.18, sigma=0.20)
        isc_anom = (cell_var - 1.2) * 0.2 + ou_process(rng, n, mu=0.0,
                                                       theta=0.20, sigma=0.20)

    else:
        raise ValueError(f"Unknown character {char}")

    # Add weekly operational cycle to everything except trajectories already
    # carrying their own cycle structure
    wc = weekly_cycle(n)
    cell_var = cell_var + wc * 0.10
    soc_max_drift = soc_max_drift + wc * 0.005
    thermal_anom = thermal_anom + wc * 0.15

    # Bounds for safety
    cell_var = np.clip(cell_var, 0.6, 6.0)
    soc_max_drift = np.clip(soc_max_drift, 0.70, 0.99)
    soc_vol = np.clip(soc_vol, 0.02, 0.35)

    return cell_var, soc_max_drift, soc_vol, thermal_anom, isc_anom


def behavioural_response(s_overcharge_arr, threshold=0.8, sustained_days=7,
                          relief_factor=0.45, relief_days=14):
    """Operator behavioural feedback loop (paper §6.1 claim made tangible):
    when s_overcharge stays above `threshold` for `sustained_days`
    consecutive days, the operator tightens the SOC envelope and the
    score drops by `relief_factor` for the next `relief_days` days.
    Returns the modified array."""
    arr = s_overcharge_arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] > threshold:
            # Count consecutive elevated days
            j = i
            while j < n and arr[j] > threshold:
                j += 1
            run_len = j - i
            if run_len >= sustained_days:
                # Apply relief starting at day j
                for k in range(j, min(n, j + relief_days)):
                    arr[k] -= arr[k] * relief_factor * \
                              (1 - (k - j) / relief_days)
                i = min(n, j + relief_days)
                continue
        i += 1
    return arr


def compute_premium(s_x: float, tiv: float = TIV_USD) -> tuple:
    """Demo pricing — see file docstring for honesty disclosure."""
    logit = BETA0_FIRE + BETA1 * s_x
    p_fire_mean = 1.0 / (1.0 + np.exp(-logit))
    p_fire_low = 1.0 / (1.0 + np.exp(-(logit - 0.15)))
    p_fire_high = 1.0 / (1.0 + np.exp(-(logit + 0.15)))

    base = 400.0
    slope = 380.0
    prem = base + slope * max(0.0, s_x)
    prem_low = prem * 0.92
    prem_high = prem * 1.10
    return float(prem), float(prem_low), float(prem_high), float(p_fire_mean), \
           float(p_fire_low), float(p_fire_high)


def regime_status(s_x: float) -> str:
    if s_x < NORMAL_MAX:
        return "NORMAL"
    elif s_x < WARNING_MAX:
        return "WARNING"
    else:
        return "CRITICAL"


def top_drivers(precursors: dict) -> list[str]:
    return sorted(precursors.keys(), key=lambda k: -abs(precursors[k]))[:3]


def generate_for_system(profile: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate 365 daily rows for one system using the precomputed
    trajectories above + behavioural feedback on s_overcharge."""
    sid = profile["system_id"]

    # SOH: monotonic decay (battery degradation IS monotonic) but with small
    # daily measurement noise
    soh_track = np.linspace(profile["soh_start"], profile["soh_end"], N_DAYS) \
                + rng.normal(0, 0.0008, N_DAYS)

    # Pre-compute trajectories
    cell_var, soc_max_drift, soc_vol, thermal_anom, isc_anom = \
        trajectory_for_character(profile["char"], rng)

    # Pre-compute s_overcharge trajectory (depends on soc_max_drift)
    s_overcharge_pre = np.maximum(0.0,
                                  (soc_max_drift - 0.85) * 8.0 +
                                  rng.normal(0, 0.20, N_DAYS))
    s_overcharge = behavioural_response(s_overcharge_pre,
                                        threshold=0.8, sustained_days=7,
                                        relief_factor=0.45, relief_days=14)

    rows = []
    for d in range(N_DAYS):
        date = DATE_START + pd.Timedelta(days=d)

        soh = soh_track[d]
        soc_max = float(np.clip(soc_max_drift[d] + rng.normal(0, soc_vol[d] * 0.3),
                                0.7, 0.99))
        soc_min = float(np.clip(0.30 + rng.normal(0, soc_vol[d] * 0.5),
                                0.05, 0.55))
        soc_mean = float((soc_max + soc_min) / 2)

        # Temperature: seasonal + system thermal anomaly
        seasonal = seasonal_temp_array(boost_c=8.0)[d]
        temp_peak = seasonal + 6.0 * thermal_anom[d] + rng.normal(0, 1.0)

        # Precursors derived from underlying trajectories
        s_ISC = float((cell_var[d] - 1.5) / 1.0 + isc_anom[d] * 0.4
                      + rng.normal(0, 0.15))
        s_thermal = float(thermal_anom[d] * 0.85 + rng.normal(0, 0.15))
        s_oc = float(s_overcharge[d])
        s_imbalance = float((cell_var[d] - 1.2) / 0.8 + rng.normal(0, 0.15))
        s_thermal_mag = float(max(0.0, s_thermal * 0.6) + rng.normal(0, 0.10))

        # Cumulative cycles
        cycles_cumul = int(d * 1.2 + rng.normal(0, 5))
        cycles_cumul = max(0, cycles_cumul)

        precursors = {
            "s_ISC": s_ISC, "s_thermal": s_thermal,
            "s_overcharge": s_oc, "s_imbalance": s_imbalance,
            "s_thermal_mag": s_thermal_mag,
        }
        s_x = sum(max(0.0, v) for v in precursors.values()) / 5.0

        prem, prem_lo, prem_hi, p_fire, p_lo, p_hi = compute_premium(s_x)

        status = regime_status(s_x)
        drivers = top_drivers(precursors)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "system_id": sid,
            "system_name": profile["system_name"],
            "soh_pct": round(soh * 100, 3),
            "soc_max_daily": round(soc_max, 4),
            "soc_min_daily": round(soc_min, 4),
            "soc_mean_daily": round(soc_mean, 4),
            "temp_peak_c": round(temp_peak, 2),
            "cell_voltage_var_mv": round(float(cell_var[d]), 3),
            "cycle_count_cumul": cycles_cumul,
            "s_ISC": round(s_ISC, 3),
            "s_thermal": round(s_thermal, 3),
            "s_overcharge": round(s_oc, 3),
            "s_imbalance": round(s_imbalance, 3),
            "s_thermal_mag": round(s_thermal_mag, 3),
            "s_x_aggregate": round(s_x, 3),
            "p_fire_7d": p_fire,
            "p_fire_low": p_lo,
            "p_fire_high": p_hi,
            "premium_weekly_usd": round(prem, 2),
            "premium_low": round(prem_lo, 2),
            "premium_high": round(prem_hi, 2),
            "regime_status": status,
            "top_driver_1": drivers[0],
            "top_driver_2": drivers[1],
            "top_driver_3": drivers[2],
        })

    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(2026)
    parts = []
    for profile in SYSTEM_PROFILES:
        df_sys = generate_for_system(profile, rng)
        parts.append(df_sys)
        n_normal = (df_sys["regime_status"] == "NORMAL").sum()
        n_warning = (df_sys["regime_status"] == "WARNING").sum()
        n_critical = (df_sys["regime_status"] == "CRITICAL").sum()
        prem_lo = df_sys["premium_weekly_usd"].min()
        prem_hi = df_sys["premium_weekly_usd"].max()
        prem_std = df_sys["premium_weekly_usd"].std()
        # Compute up/down day ratio (premium changes)
        diffs = df_sys["premium_weekly_usd"].diff().dropna()
        n_up = (diffs > 0).sum()
        n_down = (diffs < 0).sum()
        print(f"System {profile['system_id']:2d} ({profile['system_name']}):")
        print(f"  NORMAL/WARNING/CRITICAL = {n_normal}/{n_warning}/{n_critical} days")
        print(f"  Premium: ${prem_lo:.0f} - ${prem_hi:.0f}/wk, "
              f"σ=${prem_std:.0f}, up/down days = {n_up}/{n_down}")

    df = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[Saved] {OUT}")
    print(f"  Total rows: {len(df):,}  ({df['system_id'].nunique()} systems × "
          f"{df['date'].nunique()} days)")


if __name__ == "__main__":
    main()
