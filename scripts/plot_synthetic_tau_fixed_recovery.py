"""synthetic_tau_fixed_recovery_result.json から診断用の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/synthetic_tau_fixed_recovery_result.json") as f:
    d = json.load(f)

h = d["history"]
steps = np.array(h["step"])
true = d["true_params"]
BLUE = "#2a78d6"
GRAY = "#898781"

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(steps, np.clip(h["loss"], 1e-12, None), color=BLUE, linewidth=2)
ax.set_yscale("log")
ax.set_xlabel("optimization step")
ax.set_ylabel("trajectory MSE [m²] (log scale)")
ax.set_title(r"合成データ・$\tau$固定でのloss推移")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("documents/figures/synthetic_tau_fixed_loss.png")
print("saved: documents/figures/synthetic_tau_fixed_loss.png")

param_keys = ["A1", "B1", "A2", "B2"]
labels = {
    "A1": r"$A_1$ (攻撃者←守備者集団)", "B1": r"$B_1$",
    "A2": r"$A_2$ (守備者←攻撃者)", "B2": r"$B_2$",
}
fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)
for ax, key in zip(axes.flat, param_keys):
    ax.plot(steps, h[key], color=BLUE, linewidth=2, label="recovered")
    ax.axhline(true[key], color=GRAY, linestyle="--", linewidth=1, label="true")
    ax.set_title(labels[key], fontsize=11)
    ax.set_xlabel("step", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
axes.flat[0].legend(fontsize=8, loc="lower right")
fig.suptitle(r"$\tau$固定でも真値に完璧に収束(合成データ)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("documents/figures/synthetic_tau_fixed_params.png")
print("saved: documents/figures/synthetic_tau_fixed_params.png")
