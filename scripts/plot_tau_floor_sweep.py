"""tau_floor_sweep_result.json から感度分析の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/tau_floor_sweep_result.json") as f:
    d = json.load(f)

tau_mins = d["tau_min_levels"]
results = d["results"]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

rmse = [r["final_mse"] ** 0.5 for r in results]
axes[0].plot(tau_mins, rmse, marker="o", markersize=7, color=BLUE, linewidth=2)
axes[0].set_xlabel(r"$\tau$下限 [s]")
axes[0].set_ylabel("フィッティング後のRMSE [m]")
axes[0].set_title("下限を下げるほどフィットは向上し続ける")
axes[0].invert_xaxis()

tau_att = [r["final_params"]["tau_att"] for r in results]
axes[1].plot(tau_mins, tau_mins, color=GRAY, linestyle="--", linewidth=1, label="下限 = 収束値(の場合)")
axes[1].plot(tau_mins, tau_att, marker="o", markersize=7, color=ORANGE, linewidth=2, label=r"実際の収束 $\tau_{att}$")
axes[1].set_xlabel(r"$\tau$下限 [s]")
axes[1].set_ylabel(r"収束した $\tau_{att}$ [s]")
axes[1].set_title(r"$\tau$は常に下限そのものに張り付く(有限値に収束しない)")
axes[1].invert_xaxis()
axes[1].legend(fontsize=9)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("documents/figures/tau_floor_sweep.png")
print("saved: documents/figures/tau_floor_sweep.png")
