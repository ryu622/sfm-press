"""real_data_pooled_fit_v3_result.json から診断用の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/real_data_pooled_fit_v3_result.json") as f:
    d = json.load(f)

h = d["history"]
steps = np.array(h["step"])
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(steps, h["loss"], color=BLUE, linewidth=2)
ax.axhline(d["baseline_mse"], color=GRAY, linestyle="--", linewidth=1)
ax.text(len(steps) * 0.55, d["baseline_mse"] * 0.97, "ベースライン(相互作用なし)",
        fontsize=9, color=GRAY, va="top")
ax.set_xlabel("optimization step")
ax.set_ylabel("trajectory MSE [m²]  (12 windows pooled)")
ax.set_title("因果的EMA駆動力でのloss推移")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("documents/figures/real_pooled_fit_v3_loss.png")
print("saved: documents/figures/real_pooled_fit_v3_loss.png")

param_keys = ["A1", "B1", "A2", "B2", "tau_att", "tau_def"]
labels = {
    "A1": r"$A_1$ (攻撃者←守備者集団の総和)",
    "B1": r"$B_1$",
    "A2": r"$A_2$ (守備者←攻撃者)",
    "B2": r"$B_2$",
    "tau_att": r"$\tau_{att}$",
    "tau_def": r"$\tau_{def}$",
}
fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=150)
for ax, key in zip(axes.flat, param_keys):
    ax.plot(steps, h[key], color=BLUE, linewidth=2)
    ax.set_title(labels[key], fontsize=11)
    ax.set_xlabel("step", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    if key in ("tau_att", "tau_def"):
        ax.axhline(0.3, color=ORANGE, linestyle=":", linewidth=1)
fig.suptitle(r"パラメータ推移: $\tau,A_2,B_2,B_1$は収束、$\tau$は下限付近、$A_1$は緩やかに漂う",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("documents/figures/real_pooled_fit_v3_params.png")
print("saved: documents/figures/real_pooled_fit_v3_params.png")
