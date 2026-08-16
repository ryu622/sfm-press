"""tau_fixed_v4_b1_convergence_result.json から収束診断の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/tau_fixed_v4_b1_convergence_result.json") as f:
    d = json.load(f)

h = d["history"]
steps = np.array(h["step"])
BLUE = "#2a78d6"
ORANGE = "#eb6834"

fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)

axes[0].plot(steps, h["loss"], color=BLUE, linewidth=1.2)
axes[0].set_xlabel("optimization step")
axes[0].set_ylabel("trajectory MSE [m²]")
axes[0].set_title("loss: 3000ステップでも安定せず微増")

axes[1].plot(steps, h["B1"], color=ORANGE, linewidth=1.2)
axes[1].set_xlabel("optimization step")
axes[1].set_ylabel(r"$B_1$")
axes[1].set_title(r"$B_1$: 21〜305の範囲を行き来し続ける")

axes[2].plot(steps, h["A1"], color=BLUE, linewidth=1.2)
axes[2].set_xlabel("optimization step")
axes[2].set_ylabel(r"$A_1$")
axes[2].set_title(r"$A_1$: 相対的には安定(std/mean=0.16)")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle(r"$\tau$=1.2s固定・3000ステップでの$A_1,B_1$収束チェック(12局面)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("documents/figures/b1_convergence_check.png")
print("saved: documents/figures/b1_convergence_check.png")
