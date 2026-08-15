"""ema_halflife_sweep_result.json から感度分析の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/ema_halflife_sweep_result.json") as f:
    d = json.load(f)

halflifes = d["halflife_levels"]
results = d["results"]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

# --- 図1: RMSE と 改善率 ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
rmse = [r["final_mse"] ** 0.5 for r in results]
improve = [r["improvement_pct"] for r in results]

axes[0].plot(halflifes, rmse, marker="o", markersize=7, color=BLUE, linewidth=2)
axes[0].set_xlabel("EMA半減期 [s]")
axes[0].set_ylabel("フィッティング後のRMSE [m]")
axes[0].set_title("半減期を伸ばすほどフィット精度は悪化")

axes[1].plot(halflifes, improve, marker="o", markersize=7, color=ORANGE, linewidth=2)
axes[1].set_xlabel("EMA半減期 [s]")
axes[1].set_ylabel("ベースラインからの改善率 [%]")
axes[1].set_title("改善率も単調に低下")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("documents/figures/ema_halflife_sweep_fit.png")
print("saved: documents/figures/ema_halflife_sweep_fit.png")

# --- 図2: tauが常に下限に張り付くことを示す ---
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
tau_att = [r["final_params"]["tau_att"] for r in results]
tau_def = [r["final_params"]["tau_def"] for r in results]
ax.plot(halflifes, tau_att, marker="o", markersize=7, color=BLUE, linewidth=2, label=r"$\tau_{att}$")
ax.plot(halflifes, tau_def, marker="s", markersize=7, color=ORANGE, linewidth=2, label=r"$\tau_{def}$")
ax.axhline(0.3, color=GRAY, linestyle="--", linewidth=1)
ax.text(halflifes[-1] * 0.5, 0.32, "制約下限 = 0.3s", fontsize=9, color=GRAY)
ax.set_xlabel("EMA半減期 [s]")
ax.set_ylabel(r"収束した $\tau$ [s]")
ax.set_ylim(0.25, 0.5)
ax.set_title(r"$\tau$はどの半減期でも下限に張り付いたまま(仮説は不支持)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("documents/figures/ema_halflife_sweep_tau.png")
print("saved: documents/figures/ema_halflife_sweep_tau.png")
