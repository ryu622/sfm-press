"""held_out_validation_result.json からtrain/test比較の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/held_out_validation_result.json") as f:
    d = json.load(f)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

groups = ["train\n(in-sample, 30局面)", "test\n(held-out, 10局面)"]
base_rmse = [d["train_eval"]["baseline_mse"] ** 0.5, d["test_eval"]["baseline_mse"] ** 0.5]
fit_rmse = [d["train_eval"]["fitted_mse"] ** 0.5, d["test_eval"]["fitted_mse"] ** 0.5]

x = np.arange(2)
w = 0.32
axes[0].bar(x - w / 2, base_rmse, width=w, color=GRAY, label="baseline(駆動力のみ)")
axes[0].bar(x + w / 2, fit_rmse, width=w, color=BLUE, label="A,B込み")
axes[0].set_xticks(x)
axes[0].set_xticklabels(groups)
axes[0].set_ylabel("RMSE [m]")
axes[0].set_title("軌道RMSE: train vs held-out test")
axes[0].legend(fontsize=8)

improve = [d["train_eval"]["improvement_pct"], d["test_eval"]["improvement_pct"]]
axes[1].bar(x, improve, width=0.5, color=[ORANGE, BLUE])
axes[1].set_xticks(x)
axes[1].set_xticklabels(groups)
axes[1].set_ylabel("改善率 [%]")
axes[1].set_title("A,Bによる改善率: 過学習していないか")
for i, v in enumerate(improve):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)
axes[1].set_ylim(0, max(improve) * 1.3)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle(r"Held-out検証($\tau$=1.2s固定、40局面を30/10に分割)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("documents/figures/held_out_validation.png")
print("saved: documents/figures/held_out_validation.png")
