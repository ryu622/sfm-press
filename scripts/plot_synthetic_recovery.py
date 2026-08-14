"""synthetic_recovery_result.json から loss曲線・パラメータ収束の図を生成し documents/figures/ に保存する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/synthetic_recovery_result.json") as f:
    d = json.load(f)

h = d["history"]
steps = np.array(h["step"])
loss = np.array(h["loss"])
true = d["true_params"]

BLUE = "#2a78d6"
GRAY = "#52514e"

# --- 図1: loss曲線 ---
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(steps, np.clip(loss, 1e-12, None), color=BLUE, linewidth=2)
ax.set_yscale("log")
ax.set_xlabel("optimization step")
ax.set_ylabel("trajectory MSE loss (log scale)")
ax.set_title("合成データでのパラメータ復元: 最適化のloss推移")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("documents/figures/synthetic_recovery_loss.png")
print("saved: documents/figures/synthetic_recovery_loss.png")

# --- 図2: パラメータ収束(2x3グリッド) ---
param_keys = ["A1", "B1", "A2", "B2", "tau_att", "tau_def"]
labels = {
    "A1": r"$A_1$ (攻撃者←守備者集団)",
    "B1": r"$B_1$",
    "A2": r"$A_2$ (守備者←攻撃者)",
    "B2": r"$B_2$",
    "tau_att": r"$\tau_{att}$",
    "tau_def": r"$\tau_{def}$",
}

fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=150)
for ax, key in zip(axes.flat, param_keys):
    vals = np.array(h[key])
    ax.plot(steps, vals, color=BLUE, linewidth=2, label="recovered")
    ax.axhline(true[key], color=GRAY, linestyle="--", linewidth=1, label="true")
    ax.set_title(labels[key], fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("step", fontsize=9)

axes.flat[0].legend(fontsize=8, loc="lower right")
fig.suptitle("真値 vs 復元値の推移(初期値はわざと真値からずらしている)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("documents/figures/synthetic_recovery_params.png")
print("saved: documents/figures/synthetic_recovery_params.png")
