"""distance_stratified_result.json から距離帯別の改善率の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/distance_stratified_result.json") as f:
    d = json.load(f)

colors = {
    "0.1": "#2a78d6", "0.3": "#eb6834", "0.5": "#1baf7a",
    "0.8": "#eda100", "1.2": "#e87ba4",
}

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

for tau_str, bins in d["by_tau"].items():
    xs, ys = [], []
    for b in bins:
        if b["n_frames"] == 0 or b["improvement_pct"] is None:
            continue
        center = (b["lo"] + b["hi"]) / 2
        xs.append(center)
        ys.append(b["improvement_pct"])
    ax.plot(xs, ys, marker="o", markersize=6, linewidth=2, color=colors[tau_str],
             label=rf"$\tau$={tau_str}s")

ax.axhline(0, color="#898781", linestyle="-", linewidth=1)
ax.set_xlabel("最近傍守備者との距離 [m]")
ax.set_ylabel("その距離帯での改善率 [%]")
ax.set_title("距離帯別の改善率: 中距離(2-4m)で最良、最近接(<2m)ではむしろ悪化")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("documents/figures/distance_stratified.png")
print("saved: documents/figures/distance_stratified.png")
