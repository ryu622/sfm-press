"""動的な外部ポテンシャル(簡易版)による駆動力設計(real_data_fit_test.md 追試10)。

追試9で「相互作用項の定式化自体は健全、実データでの説明力不足は駆動力信号
(因果的EMA)が不完全な近似であることに起因する」と分かったことを受け、
駆動力を以下のように再設計する:

- 攻撃者: e_att(t) = シミュレーション中の現在位置から相手ゴールへの方向
  (追試3と同じ、外部情報かつ動的)。v0_attは固定値ではなく自由パラメータとして推定する
  (追試3の失敗の主因は「一定速度」という単純化だったと考え、ここを直接修正する)。
- 守備者: e_def(t) = シミュレーション中の現在位置から、攻撃者の現在の
  シミュレーション位置への方向(ボール保持者を追いかける、という最も単純な
  ピッチコントロール的direct pursuit)。v0_defも自由パラメータ。

いずれも「シミュレーション中の現在位置」から動的に計算するため、事前計算した
意図テーブル(EMA等)は不要で、選手自身の過去の軌道を一切参照しない
(リークなし)。相互作用項A1,B1,A2,B2は、この単純な直進/追跡からのズレを
説明する役割を担う。
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
GOAL_Y = PITCH_WIDTH / 2


def build_batch(df, windows: list[dict], attack_goal: dict):
    n_ep = len(windows)
    pos_obs = np.zeros((FIXED_LEN, n_ep, 1 + N_DEF, 2))
    vel0 = np.zeros((n_ep, 1 + N_DEF, 2))
    goal_pos = np.zeros((n_ep, 2))

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


def social_force_ode_v4(A1, B1, A2, B2, tau_att, tau_def, v0_att, v0_def, goal_pos):
    def f(t, state):
        pos = state[..., :2]
        vel = state[..., 2:4]
        att_pos, def_pos = pos[:, 0:1, :], pos[:, 1:, :]
        att_vel, def_vel = vel[:, 0, :], vel[:, 1:, :]

        # 攻撃者: 現在位置からゴールへの方向(動的・外部情報)
        to_goal = goal_pos - att_pos[:, 0, :]
        to_goal_dist = to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        e_att = to_goal / to_goal_dist
        drive_att = (v0_att * e_att - att_vel) / tau_att

        # 守備者: 現在位置から攻撃者の現在シミュレーション位置への方向(動的追跡)
        to_carrier = att_pos - def_pos  # (E,Ndef,2)
        to_carrier_dist = to_carrier.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        e_def = to_carrier / to_carrier_dist
        drive_def = (v0_def * e_def - def_vel) / tau_def

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


def run_experiment(df, windows: list[dict], attack_goal: dict, n_opt_steps: int = 1200,
                    verbose: bool = True, tau_min: float = TAU_MIN, tau_max: float = TAU_MAX,
                    tau_fixed: float | None = None) -> dict:
    pos_obs_t, vel0, goal_pos_t = build_batch(df, windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)

    n_ep = len(windows)
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0

    def simulate(params):
        f = social_force_ode_v4(**params, goal_pos=goal_pos_t)
        traj = odeint(f, y0, t_eval, method="rk4", options={"step_size": DT / 2})
        return traj

    fixed_tau_tensor = torch.tensor(tau_fixed) if tau_fixed is not None else None

    with torch.no_grad():
        baseline_tau = fixed_tau_tensor if fixed_tau_tensor is not None else torch.tensor(0.5)
        baseline_params = dict(
            A1=torch.tensor(0.0), B1=torch.tensor(1.0),
            A2=torch.tensor(0.0), B2=torch.tensor(1.0),
            tau_att=baseline_tau, tau_def=baseline_tau,
            v0_att=torch.tensor(4.0), v0_def=torch.tensor(2.0),
        )
        baseline_traj = simulate(baseline_params)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    if verbose:
        print(f"baseline (no interaction) MSE: {baseline_loss:.4f}  RMSE: {np.sqrt(baseline_loss):.3f} m")

    def tau_of(raw):
        return tau_min + (tau_max - tau_min) * torch.sigmoid(raw)

    raw = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
        "log_v0_att": torch.tensor(np.log(4.0), requires_grad=True),
        "log_v0_def": torch.tensor(np.log(2.0), requires_grad=True),
    }
    if fixed_tau_tensor is None:
        raw["raw_tau_att"] = torch.tensor(0.0, requires_grad=True)
        raw["raw_tau_def"] = torch.tensor(0.0, requires_grad=True)

    def get_params():
        p = dict(
            A1=raw["A1"], B1=torch.exp(raw["log_B1"]),
            A2=raw["A2"], B2=torch.exp(raw["log_B2"]),
            v0_att=torch.exp(raw["log_v0_att"]), v0_def=torch.exp(raw["log_v0_def"]),
        )
        if fixed_tau_tensor is not None:
            p["tau_att"] = fixed_tau_tensor
            p["tau_def"] = fixed_tau_tensor
        else:
            p["tau_att"] = tau_of(raw["raw_tau_att"])
            p["tau_def"] = tau_of(raw["raw_tau_def"])
        return p

    optimizer = torch.optim.Adam(raw.values(), lr=0.05)
    track_keys = ["A1", "B1", "A2", "B2", "v0_att", "v0_def", "tau_att", "tau_def"]
    history = {"step": [], "loss": [], **{k: [] for k in track_keys}}

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
        for k in track_keys:
            history[k].append(p[k])

        if verbose and (step % 40 == 0 or step == n_opt_steps - 1):
            print(f"step {step:4d}  loss={loss.item():.4f}  RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={p['A1']:+.3f} B1={p['B1']:.3f} A2={p['A2']:+.3f} B2={p['B2']:.3f}  "
                  f"v0_att={p['v0_att']:.3f} v0_def={p['v0_def']:.3f}  "
                  f"tau_att={p['tau_att']:.3f} tau_def={p['tau_def']:.3f}")

    final_params = get_params()
    with torch.no_grad():
        final_traj = simulate(final_params)
    final_loss = ((final_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    improvement = (1 - final_loss / baseline_loss) * 100
    if verbose:
        print(f"\nfinal fitted MSE: {final_loss:.4f}  RMSE: {np.sqrt(final_loss):.3f} m  "
              f"improvement: {improvement:.1f}%")

    return {
        "match_id": MATCH_ID,
        "n_windows": n_ep,
        "fixed_len": FIXED_LEN,
        "tau_fixed": tau_fixed,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {k: v.item() for k, v in final_params.items()},
        "history": history,
    }


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    out = run_experiment(df, windows, attack_goal, n_opt_steps=1200)
    with open("scripts/real_data_pooled_fit_v4_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/real_data_pooled_fit_v4_result.json")


if __name__ == "__main__":
    main()
