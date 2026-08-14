"""駆動力の目標を「外部情報ベース」に置き換えたプール推定(real_data_fit_test.md 追試3)。

これまでの実装は選手自身の観測軌道を平滑化した速度を「意図した速度」として使っており、
リーク(答えをほぼそのまま出力できてしまう近道)の懸念があった。この版では:

- 攻撃者(ボール保持者): 相手ゴール方向を e(t) とし、v0 は一定値。
  ゴール方向は「シミュレーション中の現在位置」から動的に計算するため、
  選手自身の軌道を一切参照しない(事前計算した意図テーブルも不要になった)。
- 守備者: v0=0(意図した速度なし)。実際に見せる動き(チェイス含む)は
  すべて相互作用項(社会力)側で説明させる、最も保守的な設計。

これにより駆動力の目標はどちらも観測軌道と完全に独立であり、リークの可能性を排除できる。
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from sfm_press.data import PITCH_WIDTH, determine_attacking_goal_x
from scripts.real_data_pooled_fit import find_pooled_windows, N_WINDOWS, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, N_DEF, R0, FPS, DT, build_snapshot

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_MIN = 0.3
TAU_MAX = 1.2
V0_ATT = 4.0  # 攻撃者の意図速度の大きさ [m/s](固定値)
GOAL_Y = PITCH_WIDTH / 2


def build_batch(df, windows: list[dict], attack_goal: dict):
    n_ep = len(windows)
    pos_obs = np.zeros((FIXED_LEN, n_ep, 1 + N_DEF, 2))
    vel0 = np.zeros((n_ep, 1 + N_DEF, 2))
    goal_pos = np.zeros((n_ep, 2))  # 攻撃者が向かうゴール位置(エピソードごとに固定)

    for e, w in enumerate(windows):
        entities = [w["carrier_id"]] + w["defender_ids"]
        window = df[(df["frame_id"] >= w["start"]) & (df["frame_id"] <= w["end"])]
        carrier_team = window[window["entity_id"] == w["carrier_id"]]["team_id"].iloc[0]
        period_id = int(window["period_id"].iloc[0])
        goal_pos[e] = [attack_goal[(carrier_team, period_id)], GOAL_Y]

        for i, eid in enumerate(entities):
            g = window[window["entity_id"] == eid].sort_values("frame_id")
            x, y = g["x"].to_numpy()[:FIXED_LEN], g["y"].to_numpy()[:FIXED_LEN]
            pos_obs[:, e, i, 0], pos_obs[:, e, i, 1] = x, y
            vel0[e, i] = g["vx"].iloc[0], g["vy"].iloc[0]

    return torch.tensor(pos_obs), torch.tensor(vel0), torch.tensor(goal_pos)


def social_force_ode_v2(A1, B1, A2, B2, tau_att, tau_def, goal_pos):
    def f(t, state):
        pos = state[..., :2]
        vel = state[..., 2:4]
        att_pos, def_pos = pos[:, 0:1, :], pos[:, 1:, :]
        att_vel, def_vel = vel[:, 0, :], vel[:, 1:, :]

        # 攻撃者: ゴール方向(現在のシミュレーション位置から動的に計算)へ向かう駆動力
        to_goal = goal_pos - att_pos[:, 0, :]
        to_goal_dist = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        e_att = to_goal / to_goal_dist
        drive_att = (V0_ATT * e_att - att_vel) / tau_att

        # 守備者: 意図した速度なし(純粋な減衰)
        drive_def = -def_vel / tau_def

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
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows")

    pos_obs_t, vel0, goal_pos_t = build_batch(df, windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)

    n_ep = len(windows)
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0

    def simulate(params):
        f = social_force_ode_v2(**params, goal_pos=goal_pos_t)
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
    print(f"baseline (no interaction) MSE: {baseline_loss:.4f}  RMSE: {np.sqrt(baseline_loss):.3f} m")

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
        "v0_att": V0_ATT,
        "windows": windows,
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {k: v.item() for k, v in final_params.items()},
        "history": history,
    }
    with open("scripts/real_data_pooled_fit_v2_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/real_data_pooled_fit_v2_result.json")


if __name__ == "__main__":
    main()
