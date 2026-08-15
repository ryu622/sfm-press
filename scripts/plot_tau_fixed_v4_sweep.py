"""tau_fixed_v4_sweep_result.json と tau_fixed_ab_only_result.json を比較する図を生成する。"""

import json

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/tau_fixed_v4_sweep_result.json") as f:
    v4 = json.load(f)
with open("scripts/tau_fixed_ab_only_result.json") as f:
    ema = json.load(f)

BLUE = "#2a78d6"
ORANGE = "#eb6834"

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

ema_taus = [r["tau_fixed"] for r in ema["results"]]
ema_improve = [r["improvement_pct"] for r in ema["results"]]
ax.plot(ema_taus, ema_improve, marker="o", markersize=7, color=ORANGE, linewidth=2,
         label="因果的EMA駆動力(追試7)")

v4_taus = [r["tau_fixed"] for r in v4["results"]]
v4_improve = [r["improvement_pct"] for r in v4["results"]]
ax.plot(v4_taus, v4_improve, marker="s", markersize=7, color=BLUE, linewidth=2,
         label="動的ポテンシャル駆動力(追試10)")

ax.set_xlabel(r"固定した $\tau$ [s]")
ax.set_ylabel("A,Bだけによる改善率 [%]")
ax.set_title("駆動力設計の違いによる、相互作用項の実質的な説明力の比較")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=10)
ax.set_ylim(0, 40)

fig.tight_layout()
fig.savefig("documents/figures/tau_fixed_v4_vs_ema.png")
print("saved: documents/figures/tau_fixed_v4_vs_ema.png")
