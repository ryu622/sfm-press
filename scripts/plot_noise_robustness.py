"""noise_robustness_result.json から、ノイズレベル別の相対誤差の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/noise_robustness_result.json") as f:
    d = json.load(f)

noise_levels = d["noise_levels"]
results = d["results"]
param_keys = ["A1", "B1", "A2", "B2", "tau_att", "tau_def"]
labels = {
    "A1": r"$A_1$ (攻撃者←守備者集団)",
    "B1": r"$B_1$",
    "A2": r"$A_2$ (守備者←攻撃者)",
    "B2": r"$B_2$",
    "tau_att": r"$\tau_{att}$",
    "tau_def": r"$\tau_{def}$",
}
# dataviz skillの検証済みカテゴリカルパレット(light, slot1-6)
colors = {
    "A1": "#2a78d6",
    "B1": "#eb6834",
    "A2": "#1baf7a",
    "B2": "#eda100",
    "tau_att": "#e87ba4",
    "tau_def": "#008300",
}

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for key in param_keys:
    rel_errs = [r["summary"][key]["rel_err_pct"] for r in results]
    ax.plot(noise_levels, rel_errs, marker="o", markersize=6, linewidth=2,
             color=colors[key], label=labels[key])

ax.set_xlabel("観測ノイズ標準偏差 [m]")
ax.set_ylabel("復元パラメータの相対誤差 [%]")
ax.set_title("観測ノイズに対するパラメータ復元のロバスト性")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(-0.02, max(noise_levels) + 0.02)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig("documents/figures/noise_robustness.png")
print("saved: documents/figures/noise_robustness.png")
