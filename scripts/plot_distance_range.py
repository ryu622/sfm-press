"""distance_range_result.json からヒストグラム画像を生成し、documents/figures/ に保存する。"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/distance_range_result.json") as f:
    d = json.load(f)

bin_edges = np.array(d["bin_edges"])
counts = np.array(d["counts"])
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
widths = np.diff(bin_edges)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.bar(centers, counts, width=widths * 0.9, color="#2a78d6", edgecolor="none")

for p, label in [("50", "p50"), ("90", "p90")]:
    v = d["percentiles"][p]
    ax.axvline(v, color="#52514e", linestyle="--", linewidth=1)
    ax.text(v + 0.3, ax.get_ylim()[1] * 0.95, f"{label} ({v:.1f}m)", fontsize=9, color="#52514e")

ax.set_xlabel("守備者〜ボール保持者 距離 (m)")
ax.set_ylabel("フレーム数")
ax.set_title(f"距離分布 (match {d['match_id']}, N={d['n_frames']:,} frames)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("documents/figures/defender_distance_range.png")
print("saved: documents/figures/defender_distance_range.png")
