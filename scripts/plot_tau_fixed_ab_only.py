"""tau_fixed_ab_only_result.json から診断用の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/tau_fixed_ab_only_result.json") as f:
    d = json.load(f)

taus = d["tau_fixed_levels"]
results = d["results"]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
improve_free = 36.1  # real_data_pooled_fit_v3(halflife=0.3, tau自由・下限0.3で張り付き)の改善率
improve = [r["improvement_pct"] for r in results]

ax.plot(taus, improve, marker="o", markersize=8, color=BLUE, linewidth=2, label=r"$\tau$を固定(A,Bのみ最適化)")
ax.axhline(improve_free, color=ORANGE, linestyle="--", linewidth=1.5)
ax.text(taus[0], improve_free + 1.5, r"参考: $\tau$を自由に動かした場合の改善率 (36.1%)",
        fontsize=9, color=ORANGE)

ax.set_xlabel(r"固定した $\tau$ [s]")
ax.set_ylabel("ベースラインからの改善率 [%]")
ax.set_title(r"$\tau$を固定すると、$A,B$だけの改善効果は大幅に縮小する")
ax.set_ylim(0, 40)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="center right")

fig.tight_layout()
fig.savefig("documents/figures/tau_fixed_ab_only.png")
print("saved: documents/figures/tau_fixed_ab_only.png")
