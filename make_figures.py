"""Redraw the proposal figures from saved results (no re-simulation)."""
import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import microgrid as mg

OUT = sys.argv[1]
res = pd.read_csv(f"{OUT}/policy_results.csv")
daily = json.load(open(f"{OUT}/daily_costs.json"))
train = mg.generate(3 * 365 * 24, seed=2026)

# figure 1
seg = train.iloc[24 * 30 * 9: 24 * 30 * 10].reset_index(drop=True)
days = seg.index / 24
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
in_spike = seg.regime.to_numpy() == 1
for a in ax:
    a.fill_between(days, 0, 1, where=in_spike, transform=a.get_xaxis_transform(),
                   color="#d97706", alpha=0.18, linewidth=0, label="latent spike regime (hidden from agent)")
ax[0].plot(days, seg.price, color="#1f2937", lw=0.9, label="spot price")
ax[0].plot(days, seg.fc_price, color="#2563eb", lw=0.8, alpha=0.7, label="day-ahead forecast")
ax[0].set_ylabel("$/MWh"); ax[0].legend(loc="upper center", fontsize=8, ncol=3)
ax[1].plot(days, seg.wind, color="#059669", lw=0.9, label="wind (MW)")
ax[1].plot(days, seg.fc_wind, color="#059669", lw=0.7, alpha=0.5, ls="--", label="forecast")
ax[1].set_ylabel("MW"); ax[1].legend(loc="upper center", fontsize=8, ncol=3)
ax[2].plot(days, seg.load, color="#7c3aed", lw=0.9, label="load (MW)")
ax[2].plot(days, seg.fc_load, color="#7c3aed", lw=0.7, alpha=0.5, ls="--", label="forecast")
ax[2].set_ylabel("MW"); ax[2].set_xlabel("day"); ax[2].legend(loc="upper center", fontsize=8, ncol=3)
fig.suptitle("One month of generated training data: the day-ahead forecast carries no information about spike timing; "
             "the regime is visible only in realised history", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/sample_paths.png", dpi=160)

# figure 2
fig, ax = plt.subplots(figsize=(8.5, 5.8))
colors = {"do-nothing": "#9ca3af", "greedy threshold": "#6b7280", "point-forecast MPC": "#dc2626",
          "two-stage, risk-neutral": "#f59e0b", "regime-aware CVaR (reference)": "#059669"}
for _, r in res.iterrows():
    ax.scatter(r.mean_daily_cost, r.cvar95_daily_cost, s=110, color=colors[r.policy], zorder=3,
               edgecolor="white", linewidth=1.2)
    left = r.policy in ("greedy threshold",)
    ax.annotate(f"{r.policy}\n{int(r.violations)} limit violations",
                (r.mean_daily_cost, r.cvar95_daily_cost),
                xytext=(-10 if left else 10, 14 if left else -4), textcoords="offset points",
                fontsize=8.5, va="bottom" if left else "top", ha="right" if left else "left")
ax.set_xlabel("mean daily cost ($)")
ax.set_ylabel("CVaR 95% of daily cost ($)")
ax.set_title("Expected cost vs tail cost, 40 sealed 7-day paths (lower-left is better)", fontsize=10)
ax.margins(x=0.22, y=0.12)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/policy_comparison.png", dpi=160)

# figure 3
fig, ax = plt.subplots(figsize=(8.5, 4.8))
names = list(res.policy)
data = [np.array(daily[n]) for n in names]
bp = ax.boxplot(data, tick_labels=names, showfliers=True, whis=(5, 95), patch_artist=True)
for patch, n in zip(bp["boxes"], names):
    patch.set_facecolor(colors[n]); patch.set_alpha(0.45)
ax.set_ylabel("daily cost ($)")
ax.set_title("Daily cost distribution per policy on the sealed paths (whiskers at 5th / 95th percentile)", fontsize=10)
plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/daily_cost_distribution.png", dpi=160)
print(res.to_string(index=False))
