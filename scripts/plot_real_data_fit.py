"""real_data_fit_result.json から診断用の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/real_data_fit_result.json") as f:
    d = json.load(f)

entities = d["entities"]
pos_obs = np.array(d["pos_obs"])          # (T, 4, 2)
pos_base = np.array(d["pos_pred_baseline"])
pos_fit = np.array(d["pos_pred_fitted"])
h = d["history"]
labels = ["攻撃者(ボール保持者)", "守備者1(最近傍)", "守備者2", "守備者3"]

BLUE = "#2a78d6"
GRAY = "#898781"
BLACK = "#0b0b0b"

# --- 図1: 軌道の重ね合わせ(2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
for i, ax in enumerate(axes.flat):
    ax.plot(pos_obs[:, i, 0], pos_obs[:, i, 1], color=BLACK, linewidth=2, label="観測")
    ax.plot(pos_base[:, i, 0], pos_base[:, i, 1], color=GRAY, linewidth=1.5, linestyle="--",
             label="ベースライン(相互作用なし)")
    ax.plot(pos_fit[:, i, 0], pos_fit[:, i, 1], color=BLUE, linewidth=1.5, label="フィッティング後")
    ax.scatter(*pos_obs[0, i], color=BLACK, s=30, zorder=5)
    ax.set_title(labels[i], fontsize=11)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
axes.flat[0].legend(fontsize=8, loc="best")
fig.suptitle("実データでの軌道フィッティング: 観測 vs ベースライン vs フィッティング後", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("documents/figures/real_fit_trajectories.png")
print("saved: documents/figures/real_fit_trajectories.png")

# --- 図2: loss推移 ---
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(h["step"], h["loss"], color=BLUE, linewidth=2)
ax.axhline(d["baseline_mse"], color=GRAY, linestyle="--", linewidth=1)
ax.text(len(h["step"]) * 0.6, d["baseline_mse"] * 1.03, "ベースライン(相互作用なし)",
        fontsize=9, color=GRAY)
ax.set_yscale("log")
ax.set_xlabel("optimization step")
ax.set_ylabel("trajectory MSE [m²] (log scale)")
ax.set_title("実データフィッティングのloss推移")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("documents/figures/real_fit_loss.png")
print("saved: documents/figures/real_fit_loss.png")

# --- 図3: tauの推移(縮退の懸念を示す) ---
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(h["step"], h["tau_att"], color=BLUE, linewidth=2, label=r"$\tau_{att}$")
ax.plot(h["step"], h["tau_def"], color="#eb6834", linewidth=2, label=r"$\tau_{def}$")
ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1)
ax.text(len(h["step"]) * 0.7, 0.52, "歩行者分野で典型的とされる水準(~0.5s)", fontsize=8, color=GRAY)
ax.set_xlabel("optimization step")
ax.set_ylabel(r"$\tau$ [s]")
ax.set_title(r"$\tau$の推移: 物理的に妥当な範囲を大きく下回って収束")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("documents/figures/real_fit_tau.png")
print("saved: documents/figures/real_fit_tau.png")
