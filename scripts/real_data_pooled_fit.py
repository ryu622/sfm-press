"""複数の実プレッシャー局面をプールして(B)を同時推定する(real_data_fit_test.md の追試)。

single-window fitting(real_data_fit_test.py, real_data_fit_tau_constrained.py)では
τやBが非物理的な値に発散する「劣決定」問題が確認された。この追試では、
synthetic_recovery_test.pyの20エピソード同時推定と同じ発想で、複数の実局面に
パラメータ(A1,B1,A2,B2,tau_att,tau_def)を共有させることで、単一局面固有の
癖に過剰適合しにくくなるかを検証する。τの範囲制約は前回の教訓を踏まえて維持するが、
Bへの制約は意図的に加えず、「プールする」こと自体の効果を切り分けて見る。
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from sfm_press.data import build_snapshot
from scripts.real_data_fit_test import MATCH_ID, N_DEF, R0, FPS, DT, intent_signal

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

N_WINDOWS = 12
FIXED_LEN = 150  # 6秒(25Hz)、全windowをこの長さに揃える
TAU_MIN = 0.3
TAU_MAX = 1.2


def find_pooled_windows(df, n_windows: int, fixed_len: int) -> list[dict]:
    ball = df[df["entity_id"] == "ball"][["frame_id", "period_id", "x", "y"]].rename(
        columns={"x": "bx", "y": "by"}
    )
    players = df[df["entity_id"] != "ball"].merge(ball, on=["frame_id", "period_id"])
    players["dist_to_ball"] = np.hypot(players["x"] - players["bx"], players["y"] - players["by"])

    cand = players[
        (players["ball_state"] == "alive") & (players["team_id"] == players["ball_owning_team_id"])
    ].dropna(subset=["dist_to_ball"])
    carrier_idx = cand.groupby("frame_id")["dist_to_ball"].idxmin()
    carrier = cand.loc[carrier_idx, ["frame_id", "period_id", "entity_id"]].rename(
        columns={"entity_id": "carrier_id"}
    )
    carrier = carrier.sort_values("frame_id").reset_index(drop=True)
    carrier["run_id"] = (carrier["carrier_id"] != carrier["carrier_id"].shift()).cumsum()

    runs = carrier.groupby("run_id").agg(
        carrier_id=("carrier_id", "first"),
        n=("frame_id", "size"),
        start=("frame_id", "first"),
        end=("frame_id", "last"),
    )
    runs = runs[runs["n"] >= fixed_len]

    scored = []
    for _, run in runs.iterrows():
        start, cid = int(run["start"]), run["carrier_id"]
        end = start + fixed_len - 1  # 先頭fixed_lenフレームだけを使う
        window = df[(df["frame_id"] >= start) & (df["frame_id"] <= end)]
        carrier_team = window[window["entity_id"] == cid]["team_id"].iloc[0]
        opp = window[(window["entity_id"] != "ball") & (window["team_id"] != carrier_team)]
        c_pos = window[window["entity_id"] == cid][["frame_id", "x", "y"]].rename(columns={"x": "cx", "y": "cy"})
        opp = opp.merge(c_pos, on="frame_id")
        opp["dist"] = np.hypot(opp["x"] - opp["cx"], opp["y"] - opp["cy"])
        avg_dist = opp.groupby("entity_id")["dist"].mean().sort_values()
        top = avg_dist.head(N_DEF).index.tolist()
        if len(top) < N_DEF:
            continue
        nearest = top[0]
        d = opp[opp["entity_id"] == nearest].sort_values("frame_id")["dist"]
        closing = d.iloc[0] - d.iloc[-1]
        if closing <= 0:
            continue
        scored.append(dict(start=start, end=end, carrier_id=cid, defender_ids=top, score=closing))

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:n_windows]


def build_batch(df, windows: list[dict]):
    n_ep = len(windows)
    pos_obs = np.zeros((FIXED_LEN, n_ep, 1 + N_DEF, 2))
    vel0 = np.zeros((n_ep, 1 + N_DEF, 2))
    intent = np.zeros((FIXED_LEN, n_ep, 1 + N_DEF, 2))

    for e, w in enumerate(windows):
        entities = [w["carrier_id"]] + w["defender_ids"]
        window = df[(df["frame_id"] >= w["start"]) & (df["frame_id"] <= w["end"])]
        for i, eid in enumerate(entities):
            g = window[window["entity_id"] == eid].sort_values("frame_id")
            x, y = g["x"].to_numpy()[:FIXED_LEN], g["y"].to_numpy()[:FIXED_LEN]
            pos_obs[:, e, i, 0], pos_obs[:, e, i, 1] = x, y
            vel0[e, i] = g["vx"].iloc[0], g["vy"].iloc[0]
            vx0, vy0 = intent_signal(x, y)
            intent[:, e, i, 0], intent[:, e, i, 1] = vx0, vy0

    return (
        torch.tensor(pos_obs), torch.tensor(vel0), torch.tensor(intent),
    )


def social_force_ode_batch(A1, B1, A2, B2, tau_att, tau_def, intent, t_eval):
    n_steps = intent.shape[0]
    t0, t1 = t_eval[0].item(), t_eval[-1].item()

    def f(t, state):
        idx = int(round((t.item() - t0) / (t1 - t0) * (n_steps - 1)))
        idx = max(0, min(n_steps - 1, idx))
        v0 = intent[idx]  # (E, 1+Ndef, 2)

        pos = state[..., :2]
        vel = state[..., 2:4]
        att_pos, def_pos = pos[:, 0:1, :], pos[:, 1:, :]
        att_vel, def_vel = vel[:, 0, :], vel[:, 1:, :]

        drive_att = (v0[:, 0, :] - att_vel) / tau_att
        drive_def = (v0[:, 1:, :] - def_vel) / tau_def

        diff1 = att_pos - def_pos
        dist1 = diff1.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        f_att_from_def = (A1 * torch.exp((R0 - dist1) / B1) * diff1 / dist1).sum(dim=1)

        diff2 = def_pos - att_pos
        dist2 = diff2.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        f_def_from_att = A2 * torch.exp((R0 - dist2) / B2) * diff2 / dist2

        accel_att = drive_att + f_att_from_def
        accel_def = drive_def + f_def_from_att

        d_att = torch.cat([att_vel, accel_att], dim=-1).unsqueeze(1)
        d_def = torch.cat([def_vel, accel_def], dim=-1)
        return torch.cat([d_att, d_def], dim=1)

    return f


def main():
    df = build_snapshot(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows:")
    for w in windows:
        print(f"  carrier={w['carrier_id']} start={w['start']} score={w['score']:.2f}m")

    pos_obs_t, vel0, intent_t = build_batch(df, windows)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)

    n_ep = len(windows)
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0

    def simulate(params):
        f = social_force_ode_batch(**params, intent=intent_t, t_eval=t_eval)
        traj = odeint(f, y0, t_eval, method="rk4", options={"step_size": DT / 2})
        return traj

    with torch.no_grad():
        baseline_params = dict(
            A1=torch.tensor(0.0), B1=torch.tensor(1.0),
            A2=torch.tensor(0.0), B2=torch.tensor(1.0),
            tau_att=torch.tensor(0.5), tau_def=torch.tensor(0.5),
        )
        baseline_traj = simulate(baseline_params)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    print(f"\nbaseline (no interaction) MSE: {baseline_loss:.4f}  RMSE: {np.sqrt(baseline_loss):.3f} m")

    def tau_of(raw):
        return TAU_MIN + (TAU_MAX - TAU_MIN) * torch.sigmoid(raw)

    raw = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
        "raw_tau_att": torch.tensor(0.0, requires_grad=True),
        "raw_tau_def": torch.tensor(0.0, requires_grad=True),
    }

    def get_params():
        return dict(
            A1=raw["A1"], B1=torch.exp(raw["log_B1"]),
            A2=raw["A2"], B2=torch.exp(raw["log_B2"]),
            tau_att=tau_of(raw["raw_tau_att"]), tau_def=tau_of(raw["raw_tau_def"]),
        )

    optimizer = torch.optim.Adam(raw.values(), lr=0.05)
    n_opt_steps = 1200
    history = {"step": [], "loss": [], "A1": [], "B1": [], "A2": [], "B2": [], "tau_att": [], "tau_def": []}

    for step in range(n_opt_steps):
        optimizer.zero_grad()
        params = get_params()
        traj = simulate(params)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        optimizer.step()

        p = {k: v.item() for k, v in params.items()}
        history["step"].append(step)
        history["loss"].append(loss.item())
        for k in ["A1", "B1", "A2", "B2", "tau_att", "tau_def"]:
            history[k].append(p[k])

        if step % 40 == 0 or step == n_opt_steps - 1:
            print(f"step {step:4d}  loss={loss.item():.4f}  RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={p['A1']:+.3f} B1={p['B1']:.3f} A2={p['A2']:+.3f} B2={p['B2']:.3f} "
                  f"tau_att={p['tau_att']:.3f} tau_def={p['tau_def']:.3f}")

    final_params = get_params()
    with torch.no_grad():
        final_traj = simulate(final_params)
    final_loss = ((final_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    print(f"\nfinal fitted MSE: {final_loss:.4f}  RMSE: {np.sqrt(final_loss):.3f} m")
    improvement = (1 - final_loss / baseline_loss) * 100
    print(f"improvement over no-interaction baseline: {improvement:.1f}%")

    out = {
        "match_id": MATCH_ID,
        "n_windows": n_ep,
        "fixed_len": FIXED_LEN,
        "windows": windows,
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {k: v.item() for k, v in final_params.items()},
        "history": history,
    }
    with open("scripts/real_data_pooled_fit_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/real_data_pooled_fit_result.json")


if __name__ == "__main__":
    main()
