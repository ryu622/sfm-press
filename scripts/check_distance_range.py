"""守備者-ボール保持者間の距離レンジを実データで確認する簡易チェック(事前準備段階)。

research_plan.md 5.1節の懸念(A_ij, B_ijの非識別性)を検証するための第一歩として、
実際の守備局面で観測される距離のばらつきを見る。
"""

import json

import numpy as np
from kloppy import sportec

MATCH_ID = "J03WPY"
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def main() -> None:
    tracking = sportec.load_open_tracking_data(match_id=MATCH_ID)
    df = tracking.to_df()

    team_of_player = {}
    for team in tracking.metadata.teams:
        for player in team.players:
            team_of_player[player.player_id] = team.team_id

    player_ids = sorted({c.rsplit("_", 1)[0] for c in df.columns if c.startswith("DFL-OBJ-")})

    xs = {pid: df[f"{pid}_x"].to_numpy() * PITCH_LENGTH for pid in player_ids}
    ys = {pid: df[f"{pid}_y"].to_numpy() * PITCH_WIDTH for pid in player_ids}

    ball_x = df["ball_x"].to_numpy() * PITCH_LENGTH
    ball_y = df["ball_y"].to_numpy() * PITCH_WIDTH
    owning_team = df["ball_owning_team_id"].to_numpy()
    ball_state = df["ball_state"].to_numpy()

    n = len(df)
    nearest_defender_dist = np.full(n, np.nan)

    teams = list({t for t in team_of_player.values()})

    for team_id in teams:
        mask = (owning_team == team_id) & (ball_state == "alive")
        if not mask.any():
            continue

        attackers = [pid for pid in player_ids if team_of_player.get(pid) == team_id]
        defenders = [pid for pid in player_ids if team_of_player.get(pid) != team_id]

        # ボール保持チーム内で最もボールに近い選手 = ボール保持者(近似)
        att_dists = np.stack(
            [np.hypot(xs[pid] - ball_x, ys[pid] - ball_y) for pid in attackers], axis=1
        )
        att_dists_masked = np.where(mask[:, None], att_dists, np.inf)
        carrier_idx = np.nanargmin(att_dists_masked, axis=1)

        carrier_x = np.choose(carrier_idx, [xs[pid] for pid in attackers])
        carrier_y = np.choose(carrier_idx, [ys[pid] for pid in attackers])

        def_dists = np.stack(
            [np.hypot(xs[pid] - carrier_x, ys[pid] - carrier_y) for pid in defenders], axis=1
        )
        nearest_def = np.nanmin(def_dists, axis=1)

        nearest_defender_dist[mask] = nearest_def[mask]

    valid = nearest_defender_dist[~np.isnan(nearest_defender_dist)]

    print(f"frames analyzed (ball alive, owning team known): {len(valid)} / {n}")
    print()
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("nearest-defender-to-ball-carrier distance [m]")
    for p in percentiles:
        print(f"  p{p:>2}: {np.percentile(valid, p):6.2f} m")
    print(f"  min : {valid.min():6.2f} m")
    print(f"  max : {valid.max():6.2f} m")
    print(f"  mean: {valid.mean():6.2f} m  std: {valid.std():6.2f} m")

    # 「プレッシャー局面」の目安として近距離帯(<5m)がどれくらいの割合を占めるか
    for thresh in [2, 3, 5, 8, 10]:
        frac = (valid < thresh).mean()
        print(f"  frac(dist < {thresh}m): {frac:.3f}")

    bin_edges = np.arange(0, 42 + 1, 1.0)
    counts, _ = np.histogram(valid, bins=bin_edges)
    out = {
        "match_id": MATCH_ID,
        "n_frames": int(len(valid)),
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "percentiles": {str(p): float(np.percentile(valid, p)) for p in percentiles},
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
    }
    with open("scripts/distance_range_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved: scripts/distance_range_result.json")


if __name__ == "__main__":
    main()
