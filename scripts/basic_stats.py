"""前処理済みスナップショットの基本統計量・可視化(phase1_data_pipeline.md 用)。"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kloppy import sportec
from mplsoccer import Pitch

from sfm_press.data import PITCH_LENGTH, PITCH_WIDTH, build_snapshot

plt.rcParams["font.family"] = "Hiragino Sans"

MATCH_ID = "J03WPY"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

df = build_snapshot(MATCH_ID)
tracking_meta = sportec.load_open_tracking_data(match_id=MATCH_ID, limit=1).metadata
team_name = {t.team_id: t.name for t in tracking_meta.teams}

players = df[df["entity_id"] != "ball"].copy()
players["speed"] = np.hypot(players["vx"], players["vy"])

# ============================================================
# 1. 速度分布ヒストグラム
# ============================================================
speed_valid = players["speed"].dropna()
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.hist(speed_valid, bins=np.arange(0, 12.5, 0.25), color=BLUE, edgecolor="none")
for p, label in [(50, "p50"), (95, "p95"), (99, "p99")]:
    v = np.percentile(speed_valid, p)
    ax.axvline(v, color="#52514e", linestyle="--", linewidth=1)
    ax.text(v + 0.1, ax.get_ylim()[1] * 0.9, f"{label}\n{v:.1f}", fontsize=8, color="#52514e")
ax.set_xlabel("選手の速さ [m/s]")
ax.set_ylabel("フレーム数")
ax.set_title(f"選手の速度分布 (match {MATCH_ID})")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("documents/figures/basic_speed_hist.png")
print("saved: documents/figures/basic_speed_hist.png")

# ============================================================
# 2. 選手位置のヒートマップ(全選手・全フレーム)
# ============================================================
pos_valid = players.dropna(subset=["x", "y"])
pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
              pitch_color="#fcfcfb", line_color="#c3c2b7")
fig, ax = pitch.draw(figsize=(10, 7))
bin_stat = pitch.bin_statistic(pos_valid["x"], pos_valid["y"], statistic="count", bins=(35, 23))
bin_stat["statistic"] = np.log1p(bin_stat["statistic"])  # 密集エリアが飽和しないよう対数化
pitch.heatmap(bin_stat, ax=ax, cmap="Blues", alpha=0.85, zorder=0.5)
ax.set_title(f"選手位置の密度ヒートマップ (match {MATCH_ID}, log scale)", fontsize=12)
fig.savefig("documents/figures/basic_position_heatmap.png", dpi=150, bbox_inches="tight")
print("saved: documents/figures/basic_position_heatmap.png")

# ============================================================
# 3. ボール保持時間の割合(チーム別)
# ============================================================
alive = df[(df["entity_id"] == "ball") & (df["ball_state"] == "alive")].copy()
possession_counts = alive["ball_owning_team_id"].value_counts()
possession_frac = possession_counts / possession_counts.sum()

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
names = [team_name.get(tid, tid) for tid in possession_frac.index]
colors = [BLUE, ORANGE]
bars = ax.bar(names, possession_frac.values * 100, color=colors[: len(names)], width=0.5)
for bar, v in zip(bars, possession_frac.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 1, f"{v * 100:.1f}%",
            ha="center", fontsize=10, color="#0b0b0b")
ax.set_ylabel("ボール保持時間の割合 [%]")
ax.set_title(f"チーム別ボール保持率 (match {MATCH_ID})")
ax.set_ylim(0, max(possession_frac.values) * 100 + 10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("documents/figures/basic_possession_share.png")
print("saved: documents/figures/basic_possession_share.png")

# ============================================================
# 4. フル出場選手の走行距離ランキング
# ============================================================
full_players = players.groupby("entity_id").filter(lambda g: g["x"].isna().mean() == 0)
dt = 1.0 / tracking_meta.frame_rate
dist_km = (full_players.groupby("entity_id")["speed"].sum() * dt) / 1000.0
team_of_entity = full_players.groupby("entity_id")["team_id"].first()
dist_df = pd.DataFrame({"distance_km": dist_km, "team_id": team_of_entity}).sort_values(
    "distance_km", ascending=True
)

fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
bar_colors = [BLUE if tid == list(team_name.keys())[0] else ORANGE for tid in dist_df["team_id"]]
ax.barh(range(len(dist_df)), dist_df["distance_km"], color=bar_colors)
ax.set_yticks(range(len(dist_df)))
ax.set_yticklabels([f"{eid[-6:]}" for eid in dist_df.index], fontsize=7)
ax.set_xlabel("走行距離 [km]")
ax.set_title(f"フル出場選手の走行距離 (match {MATCH_ID}, N={len(dist_df)})", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
from matplotlib.patches import Patch
legend_elems = [Patch(color=BLUE, label=team_name.get(list(team_name.keys())[0])),
                Patch(color=ORANGE, label=team_name.get(list(team_name.keys())[1]))]
ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("documents/figures/basic_distance_ranking.png")
print("saved: documents/figures/basic_distance_ranking.png")

# ============================================================
# サマリーJSON
# ============================================================
summary = {
    "match_id": MATCH_ID,
    "speed_percentiles": {str(p): float(np.percentile(speed_valid, p)) for p in [50, 90, 95, 99]},
    "speed_max": float(speed_valid.max()),
    "possession_share": {team_name.get(tid, tid): float(f) for tid, f in possession_frac.items()},
    "distance_km": {eid: float(v) for eid, v in dist_km.items()},
}
with open("scripts/basic_stats_result.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("saved: scripts/basic_stats_result.json")
