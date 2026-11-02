"""Generate training data, evaluate five policies on sealed paths, make the figures."""
import sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import microgrid as mg

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N_PATHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
PATH_DAYS = 7
HORIZON = 24
TRAIN_HOURS = 3 * 365 * 24

# ---------------------------------------------------------------- data
train = mg.generate(TRAIN_HOURS, seed=2026)
agent_train = train.drop(columns=["regime"])
agent_train.to_csv(f"{OUT}/train.csv", index=False)

rng = np.random.default_rng(9001)
paths = []
for i in range(N_PATHS):
    seed = int(rng.integers(1, 2**31))
    start = int(rng.integers(0, 365 * 24))
    regime0 = int(rng.random() < 0.08)
    paths.append(mg.generate(PATH_DAYS * 24 + HORIZON, seed=seed, start_hour=start, regime0=regime0))

print(f"train rows {len(train)}, spike share {train.regime.mean():.3f}, "
      f"mean price {train.price.mean():.1f}, p99 {np.quantile(train.price, .99):.0f}")


def evaluate(policy_cls):
    pol = policy_cls()
    pol.calibrate(agent_train)
    t0 = time.time()
    daily_all, viol = [], 0
    for p in paths:
        d, v = mg.replay(pol, p, horizon=HORIZON)
        daily_all.append(d); viol += v
    daily = np.concatenate(daily_all)
    return {
        "policy": pol.name,
        "mean_daily_cost": float(daily.mean()),
        "cvar95_daily_cost": mg.cvar(daily, 0.95),
        "worst_day": float(daily.max()),
        "violations": int(viol),
        "seconds": round(time.time() - t0, 1),
        "daily": daily.tolist(),
    }


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(evaluate, mg.POLICIES))
    for r in results:
        print(f"{r['policy']:32s} mean {r['mean_daily_cost']:9.0f}  CVaR95 {r['cvar95_daily_cost']:9.0f}  "
              f"worst {r['worst_day']:9.0f}  viol {r['violations']:4d}  {r['seconds']}s")
    pd.DataFrame([{k: v for k, v in r.items() if k != "daily"} for r in results]) \
        .to_csv(f"{OUT}/policy_results.csv", index=False)
    with open(f"{OUT}/daily_costs.json", "w") as f:
        json.dump({r["policy"]: r["daily"] for r in results}, f)

    # ---------------------------------------------------------------- figure 1: sample paths
    seg = train.iloc[24 * 30 * 9: 24 * 30 * 10].reset_index(drop=True)   # one autumn month
    days = seg.index / 24
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for a in ax:
        in_spike = seg.regime.to_numpy() == 1
        a.fill_between(days, 0, 1, where=in_spike, transform=a.get_xaxis_transform(),
                       color="#d97706", alpha=0.18, linewidth=0, label="latent spike regime (hidden)")
    ax[0].plot(days, seg.price, color="#1f2937", lw=0.9, label="spot price")
    ax[0].plot(days, seg.fc_price, color="#2563eb", lw=0.8, alpha=0.7, label="day-ahead forecast")
    ax[0].set_ylabel("$/MWh"); ax[0].legend(loc="upper left", fontsize=8, ncol=3)
    ax[1].plot(days, seg.wind, color="#059669", lw=0.9, label="wind (MW)")
    ax[1].plot(days, seg.fc_wind, color="#059669", lw=0.7, alpha=0.5, ls="--", label="forecast")
    ax[1].set_ylabel("MW"); ax[1].legend(loc="upper left", fontsize=8, ncol=2)
    ax[2].plot(days, seg.load, color="#7c3aed", lw=0.9, label="load (MW)")
    ax[2].plot(days, seg.fc_load, color="#7c3aed", lw=0.7, alpha=0.5, ls="--", label="forecast")
    ax[2].set_ylabel("MW"); ax[2].set_xlabel("day"); ax[2].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("One month of generated data: the forecast sees only part of each price spike", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sample_paths.png", dpi=160)

    # ---------------------------------------------------------------- figure 2: policy comparison
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = ["#9ca3af", "#6b7280", "#dc2626", "#f59e0b", "#059669"]
    for r, col in zip(results, colors):
        ax.scatter(r["mean_daily_cost"], r["cvar95_daily_cost"], s=90, color=col, zorder=3)
        ax.annotate(r["policy"], (r["mean_daily_cost"], r["cvar95_daily_cost"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("mean daily cost ($)")
    ax.set_ylabel("CVaR 95% of daily cost ($)")
    ax.set_title(f"Expected cost vs tail cost on {N_PATHS} sealed 7-day paths (lower-left is better)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/policy_comparison.png", dpi=160)

    # ---------------------------------------------------------------- figure 3: daily cost distributions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [np.array(r["daily"]) for r in results]
    ax.boxplot(data, tick_labels=[r["policy"] for r in results], showfliers=True, whis=(5, 95))
    ax.set_ylabel("daily cost ($)")
    ax.set_title("Daily cost distribution per policy (whiskers at 5th/95th percentile)", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/daily_cost_distribution.png", dpi=160)
    print("figures written")
