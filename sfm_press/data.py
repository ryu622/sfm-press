"""idsse-dataの前処理パイプライン(research_plan.md 6.2節)。

kloppy経由でトラッキングデータをロードし、
- 全選手(GK除く)+ボールの座標をロング形式のスナップショットに整形(メートル換算)
- Savitzky-Golayフィルタで速度・加速度を算出
- 前処理結果を .parquet でキャッシュ

を行う。イベントデータとの紐付け(正式なボール保持者特定)・トランジション局面の扱い・
train/valid/test分割はフェーズ1の対象外とし、複数試合を扱う段階で別途実装する。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kloppy import sportec
from scipy.signal import savgol_filter

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# idsse-data(6.2節)で利用可能な試合ID
MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]

DEFAULT_CACHE_DIR = Path("data/cache")

SMOOTH_WINDOW = 9      # フレーム数(25Hzで0.36秒)
SMOOTH_POLYORDER = 3


def _goalkeeper_ids(teams) -> set[str]:
    return {
        p.player_id
        for team in teams
        for p in team.players
        if str(p.starting_position) == "Goalkeeper"
    }


def _team_of_player(teams) -> dict[str, str]:
    return {p.player_id: team.team_id for team in teams for p in team.players}


def _smooth_diff(x: np.ndarray, dt: float, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """短い欠損(<=5フレーム)は線形補間、それ以上の欠損はNaNのまま伝播させる。"""
    s = pd.Series(x).interpolate(limit=5).to_numpy()
    n_valid = np.count_nonzero(~np.isnan(s))
    if n_valid < window:
        nan = np.full_like(s, np.nan)
        return nan, nan.copy(), nan.copy()

    # NaNが残る場合、そのまま渡すとsavgol_filterがエラーになるため一時的に0で埋め、
    # 有効フレームからwindow//2以内にあるサンプルだけ後でNaNに戻す。
    isnan = np.isnan(s)
    filled = np.where(isnan, 0.0, s)

    pos = savgol_filter(filled, window, polyorder, deriv=0)
    vel = savgol_filter(filled, window, polyorder, deriv=1, delta=dt)
    acc = savgol_filter(filled, window, polyorder, deriv=2, delta=dt)

    if isnan.any():
        half = window // 2
        near_nan = pd.Series(isnan).rolling(window=2 * half + 1, center=True, min_periods=1).max().astype(bool).to_numpy()
        pos = np.where(near_nan, np.nan, pos)
        vel = np.where(near_nan, np.nan, vel)
        acc = np.where(near_nan, np.nan, acc)

    return pos, vel, acc


def _build_long_snapshot(match_id: str) -> pd.DataFrame:
    tracking = sportec.load_open_tracking_data(match_id=match_id)
    df = tracking.to_df()
    dt = 1.0 / tracking.metadata.frame_rate

    teams = tracking.metadata.teams
    gk_ids = _goalkeeper_ids(teams)
    team_of = _team_of_player(teams)

    player_ids = sorted(
        {c.rsplit("_", 1)[0] for c in df.columns if c.startswith("DFL-OBJ-")} - gk_ids
    )

    meta_cols = df[["frame_id", "period_id", "timestamp", "ball_state", "ball_owning_team_id"]].copy()
    meta_cols["timestamp"] = meta_cols["timestamp"].dt.total_seconds()

    entities = []

    ball = meta_cols.copy()
    ball["entity_id"] = "ball"
    ball["team_id"] = None
    ball["x"] = df["ball_x"].to_numpy() * PITCH_LENGTH
    ball["y"] = df["ball_y"].to_numpy() * PITCH_WIDTH
    entities.append(ball)

    for pid in player_ids:
        if f"{pid}_x" not in df.columns:
            continue
        e = meta_cols.copy()
        e["entity_id"] = pid
        e["team_id"] = team_of.get(pid)
        e["x"] = df[f"{pid}_x"].to_numpy() * PITCH_LENGTH
        e["y"] = df[f"{pid}_y"].to_numpy() * PITCH_WIDTH
        entities.append(e)

    long_df = pd.concat(entities, ignore_index=True)
    long_df = long_df.sort_values(["entity_id", "period_id", "frame_id"]).reset_index(drop=True)

    out_parts = []
    for (_entity_id, _period_id), g in long_df.groupby(["entity_id", "period_id"], sort=False):
        g = g.sort_values("frame_id")
        pos_x, vx, ax = _smooth_diff(g["x"].to_numpy(), dt, SMOOTH_WINDOW, SMOOTH_POLYORDER)
        pos_y, vy, ay = _smooth_diff(g["y"].to_numpy(), dt, SMOOTH_WINDOW, SMOOTH_POLYORDER)
        g = g.copy()
        g["x"], g["y"] = pos_x, pos_y
        g["vx"], g["vy"] = vx, vy
        g["ax"], g["ay"] = ax, ay
        out_parts.append(g)

    result = pd.concat(out_parts, ignore_index=True)
    result = result.sort_values(["frame_id", "entity_id"]).reset_index(drop=True)
    return result


def build_snapshot(match_id: str, cache_dir: Path = DEFAULT_CACHE_DIR, force: bool = False) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{match_id}.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    result = _build_long_snapshot(match_id)

    cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    return result
