"""tau_fixed_v4_fair_result.json と過去の結果を比較する図を生成する。"""

import json

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"

with open("scripts/tau_fixed_v4_fair_result.json") as f:
    fair = json.load(f)
with open("scripts/tau_fixed_ab_only_result.json") as f:
    ema = json.load(f)
with open("scripts/tau_fixed_v4_sweep_result.json") as f:
    unfair = json.load(f)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)

ema_taus = [r["tau_fixed"] for r in ema["results"]]
ema_improve = [r["improvement_pct"] for r in ema["results"]]
ax.plot(ema_taus, ema_improve, marker="o", markersize=7, color=ORANGE, linewidth=2,
         label="因果的EMA駆動力(追試7)")

unfair_taus = [r["tau_fixed"] for r in unfair["results"]]
unfair_improve = [r["improvement_pct"] for r in unfair["results"]]
ax.plot(unfair_taus, unfair_improve, marker="^", markersize=7, color=GRAY, linewidth=1.5,
         linestyle=":", label="動的ポテンシャル(v0も同時に自由、不公平な比較)")

fair_taus = [r["tau_fixed"] for r in fair["results"]]
fair_improve = [r["improvement_pct"] for r in fair["results"]]
ax.plot(fair_taus, fair_improve, marker="s", markersize=7, color=BLUE, linewidth=2,
         label="動的ポテンシャル(v0固定・フェア比較, 追試11)")

ax.set_xlabel(r"固定した $\tau$ [s]")
ax.set_ylabel("A,Bだけによる改善率 [%]")
ax.set_title("駆動力設計の比較(フェア版 vs 不公平版)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
ax.set_ylim(0, 40)

fig.tight_layout()
fig.savefig("documents/figures/tau_fixed_v4_fair_vs_unfair.png")
print("saved: documents/figures/tau_fixed_v4_fair_vs_unfair.png")
