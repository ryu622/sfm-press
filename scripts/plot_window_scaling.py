"""12局面(追試12)と40局面(追試13)でのB1,A1収束の違いを比較する図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/tau_fixed_v4_b1_convergence_result.json") as f:
    d12 = json.load(f)
with open("scripts/window_scaling_test_result.json") as f:
    d40 = json.load(f)

BLUE = "#2a78d6"
ORANGE = "#eb6834"

fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)

h12, h40 = d12["history"], d40["history"]
steps12, steps40 = np.array(h12["step"]), np.array(h40["step"])

axes[0, 0].plot(steps12, h12["B1"], color=ORANGE, linewidth=1.0)
axes[0, 0].set_title("12局面: $B_1$は20〜300超を振動")
axes[0, 0].set_ylabel(r"$B_1$")

axes[0, 1].plot(steps40, h40["B1"], color=BLUE, linewidth=1.0)
axes[0, 1].set_title(r"40局面: $B_1$は200ステップ程度で収束")

axes[1, 0].plot(steps12, h12["A1"], color=ORANGE, linewidth=1.0)
axes[1, 0].set_title(r"12局面: $A_1$(相対的には安定だが緩やかに漂う)")
axes[1, 0].set_ylabel(r"$A_1$")
axes[1, 0].set_xlabel("optimization step")

axes[1, 1].plot(steps40, h40["A1"], color=BLUE, linewidth=1.0)
axes[1, 1].set_title(r"40局面: $A_1$も同様に収束")
axes[1, 1].set_xlabel("optimization step")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle(r"局面数増加による$A_1,B_1$の識別性改善($\tau$=1.2s固定, 段階2=3000ステップ)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("documents/figures/window_scaling_convergence.png")
print("saved: documents/figures/window_scaling_convergence.png")
