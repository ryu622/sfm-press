"""real_data_fit_tau_constrained_result.json から診断用の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/real_data_fit_tau_constrained_result.json") as f:
    d = json.load(f)

entities = d["entities"]
pos_obs = np.array(d["pos_obs"])
pos_base = np.array(d["pos_pred_baseline"])
pos_fit = np.array(d["pos_pred_fitted"])
h = d["history"]
labels = ["攻撃者(ボール保持者)", "守備者1(最近傍)", "守備者2", "守備者3"]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"
BLACK = "#0b0b0b"

# --- 図1: 軌道の重ね合わせ ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
for i, ax in enumerate(axes.flat):
    ax.plot(pos_obs[:, i, 0], pos_obs[:, i, 1], color=BLACK, linewidth=2, label="観測")
    ax.plot(pos_base[:, i, 0], pos_base[:, i, 1], color=GRAY, linewidth=1.5, linestyle="--",
             label="ベースライン(相互作用なし)")
    ax.plot(pos_fit[:, i, 0], pos_fit[:, i, 1], color=BLUE, linewidth=1.5, label="フィッティング後(τ制約あり)")
    ax.scatter(*pos_obs[0, i], color=BLACK, s=30, zorder=5)
    ax.set_title(labels[i], fontsize=11)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
axes.flat[0].legend(fontsize=8, loc="best")
fig.suptitle("τ制約下での軌道フィッティング: 観測 vs ベースライン vs フィッティング後", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("documents/figures/real_fit_tau_constrained_trajectories.png")
print("saved: documents/figures/real_fit_tau_constrained_trajectories.png")

# --- 図2: B1, B2の発散 ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
axes[0].plot(h["step"], h["B1"], color=BLUE, linewidth=2, label=r"$B_1$")
axes[0].plot(h["step"], h["B2"], color=ORANGE, linewidth=2, label=r"$B_2$")
axes[0].set_xlabel("optimization step")
axes[0].set_ylabel(r"$B$ [m]")
axes[0].set_title(r"$B_1, B_2$: 収束せず発散し続けている")
axes[0].legend(fontsize=9)

axes[1].plot(h["step"], h["tau_att"], color=BLUE, linewidth=2, label=r"$\tau_{att}$")
axes[1].plot(h["step"], h["tau_def"], color=ORANGE, linewidth=2, label=r"$\tau_{def}$")
axes[1].axhline(d["tau_min"], color=GRAY, linestyle="--", linewidth=1)
axes[1].text(len(h["step"]) * 0.5, d["tau_min"] + 0.02, f"下限={d['tau_min']}s", fontsize=8, color=GRAY)
axes[1].set_xlabel("optimization step")
axes[1].set_ylabel(r"$\tau$ [s]")
axes[1].set_title(r"$\tau$: 制約により妥当な範囲に収まって安定")
axes[1].legend(fontsize=9)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("documents/figures/real_fit_tau_constrained_params.png")
print("saved: documents/figures/real_fit_tau_constrained_params.png")
