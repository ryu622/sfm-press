"""駆動力の目標を「因果的な短時間平滑化」に置き換えたプール推定(real_data_fit_test.md 追試4)。

追試1-3で試した2つの極端(自己参照・中心化2秒窓 vs 完全外部固定)の中間案。
過去のみを参照する(未来のフレームを使わない)短い指数移動平均(EMA)で
選手自身の速度方向・大きさを推定し、それを意図した速度v0*e(t)とする。

- 未来を参照しないためリークが構造的に起きない(等速直線運動モデルの標準的な近似)
- 中心化長期平滑化と異なり、選手の実際の方向転換に遅れて追従するため、
  そのズレを説明する仕事がtau・相互作用項に残る
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from scripts.real_data_pooled_fit import find_pooled_windows, N_WINDOWS, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, N_DEF, R0, FPS, DT, build_snapshot

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_MIN = 0.3
TAU_MAX = 1.2
EMA_HALFLIFE_SEC = 0.3  # 因果的EMAの半減期


def causal_ema_velocity(vx: np.ndarray, vy: np.ndarray, dt: float, halflife_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """観測速度(vx, vy)に対する因果的(過去のみ参照)な指数移動平均。"""
    alpha = 1 - 0.5 ** (dt / halflife_sec)
    ema_x = np.empty_like(vx)
    ema_y = np.empty_like(vy)
    ema_x[0], ema_y[0] = vx[0], vy[0]
    for t in range(1, len(vx)):
        ema_x[t] = alpha * vx[t] + (1 - alpha) * ema_x[t - 1]
        ema_y[t] = alpha * vy[t] + (1 - alpha) * ema_y[t - 1]
    return ema_x, ema_y


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
            vx, vy = g["vx"].to_numpy()[:FIXED_LEN], g["vy"].to_numpy()[:FIXED_LEN]
            pos_obs[:, e, i, 0], pos_obs[:, e, i, 1] = x, y
            vel0[e, i] = vx[0], vy[0]
            ema_vx, ema_vy = causal_ema_velocity(vx, vy, DT, EMA_HALFLIFE_SEC)
            intent[:, e, i, 0], intent[:, e, i, 1] = ema_vx, ema_vy

    return torch.tensor(pos_obs), torch.tensor(vel0), torch.tensor(intent)


def social_force_ode_v3(A1, B1, A2, B2, tau_att, tau_def, intent, t_eval):
    n_steps = intent.shape[0]
    t0, t1 = t_eval[0].item(), t_eval[-1].item()

    def f(t, state):
        idx = int(round((t.item() - t0) / (t1 - t0) * (n_steps - 1)))
        idx = max(0, min(n_steps - 1, idx))
        v0 = intent[idx]  # (E, 1+Ndef, 2) 因果的EMA速度(方向・大きさ込み)

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
    print(f"selected {len(windows)} windows")

    pos_obs_t, vel0, intent_t = build_batch(df, windows)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)

    n_ep = len(windows)
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0

    def simulate(params):
        f = social_force_ode_v3(**params, intent=intent_t, t_eval=t_eval)
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
        "ema_halflife_sec": EMA_HALFLIFE_SEC,
        "windows": windows,
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {k: v.item() for k, v in final_params.items()},
        "history": history,
    }
    with open("scripts/real_data_pooled_fit_v3_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/real_data_pooled_fit_v3_result.json")


if __name__ == "__main__":
    main()
