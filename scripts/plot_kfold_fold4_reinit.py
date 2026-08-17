"""kfold_cv_result.jsonとkfold_fold4_reinit_result.jsonから、fold4補正の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/kfold_cv_result.json") as f:
    kfold = json.load(f)
with open("scripts/kfold_fold4_reinit_result.json") as f:
    reinit = json.load(f)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"
RED = "#c0392b"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

# 左: 元の5分割の結果(fold4を赤で強調)
folds = [r["fold"] for r in kfold["results"]]
improves = [r["test_improve"] for r in kfold["results"]]
colors = [RED if f == 4 else BLUE for f in folds]
axes[0].bar([f"fold={f}" for f in folds], improves, color=colors)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("held-out改善率 [%]")
axes[0].set_title("元の5分割CV結果(fold=4が破綻)")

# 右: fold=4の初期値違いでの再実行結果
labels = [r["init_label"] for r in reinit["results"]]
train_rmse = [r["train_mse"] ** 0.5 for r in reinit["results"]]
test_improve = [r["test_improve"] for r in reinit["results"]]
colors2 = [RED if lbl.startswith("original") else BLUE for lbl in labels]

ax2 = axes[1]
x = np.arange(len(labels))
ax2.bar(x, test_improve, color=colors2)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([f"train\nRMSE={v:.2f}m" for v in train_rmse], fontsize=8)
ax2.set_ylabel("held-out改善率 [%]")
ax2.set_title("fold=4を初期値違いで再実行(赤=元の初期値)")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle("fold=4の破綻は初期値依存の局所解だった", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("documents/figures/kfold_fold4_reinit.png")
print("saved: documents/figures/kfold_fold4_reinit.png")
