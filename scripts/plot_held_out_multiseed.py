"""held_out_validation_multiseed_result.json から5シード分のtrain/test改善率の図を生成する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/held_out_validation_multiseed_result.json") as f:
    d = json.load(f)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

seeds = d["seeds"]
train_vals = [r["train_improve"] for r in d["results"]]
test_vals = [r["test_improve"] for r in d["results"]]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
x = np.arange(len(seeds))
w = 0.35
ax.bar(x - w / 2, train_vals, width=w, color=ORANGE, label="train(in-sample, 30局面)")
ax.bar(x + w / 2, test_vals, width=w, color=BLUE, label="test(held-out, 10局面)")

ax.axhline(np.mean(train_vals), color=ORANGE, linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(np.mean(test_vals), color=BLUE, linestyle="--", linewidth=1, alpha=0.7)
ax.text(4.55, np.mean(train_vals), f"train平均{np.mean(train_vals):.1f}%", color=ORANGE, fontsize=8, va="bottom")
ax.text(4.55, np.mean(test_vals), f"test平均{np.mean(test_vals):.1f}%", color=BLUE, fontsize=8, va="top")

ax.set_xticks(x)
ax.set_xticklabels([f"seed={s}" for s in seeds])
ax.set_ylabel("改善率 [%]")
ax.set_title("5つの乱数分割でのtrain/test改善率(τ=1.2s固定)")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlim(-0.6, 5.3)

fig.tight_layout()
fig.savefig("documents/figures/held_out_multiseed.png")
print("saved: documents/figures/held_out_multiseed.png")
