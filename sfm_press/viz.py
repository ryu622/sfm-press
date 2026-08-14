"""前処理済みスナップショットの可視化(デバッグ用、research_plan.md フェーズ1)。"""

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

from sfm_press.data import PITCH_LENGTH, PITCH_WIDTH

TEAM_COLORS = ["#2a78d6", "#eb6834"]  # dataviz skillのカテゴリカルパレット slot1/2
BALL_COLOR = "#0b0b0b"


def plot_frame(snapshot: pd.DataFrame, frame_id: int, show_velocity: bool = True, save_path: str | None = None):
    frame = snapshot[snapshot["frame_id"] == frame_id]
    if frame.empty:
        raise ValueError(f"frame_id={frame_id} not found in snapshot")

    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                  pitch_color="#fcfcfb", line_color="#c3c2b7")
    fig, ax = pitch.draw(figsize=(10, 7))

    team_ids = sorted(frame.loc[frame["entity_id"] != "ball", "team_id"].dropna().unique())
    team_color = {tid: TEAM_COLORS[i % len(TEAM_COLORS)] for i, tid in enumerate(team_ids)}

    players = frame[frame["entity_id"] != "ball"].dropna(subset=["x", "y"])
    for tid, color in team_color.items():
        sub = players[players["team_id"] == tid]
        pitch.scatter(sub["x"], sub["y"], ax=ax, color=color, s=200, edgecolors="white",
                      linewidth=1.5, zorder=3, label=tid)

    if show_velocity:
        for _, row in players.iterrows():
            if pd.notna(row["vx"]) and pd.notna(row["vy"]):
                ax.arrow(row["x"], row["y"], row["vx"] * 0.5, row["vy"] * 0.5,
                          head_width=0.8, head_length=0.8, fc="#52514e", ec="#52514e",
                          zorder=4, alpha=0.7)

    ball = frame[frame["entity_id"] == "ball"]
    if not ball.empty and pd.notna(ball["x"].iloc[0]):
        pitch.scatter(ball["x"], ball["y"], ax=ax, color=BALL_COLOR, s=60, zorder=5)

    ax.set_title(f"frame_id={frame_id}", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
