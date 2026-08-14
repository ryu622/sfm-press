"""実データでの(B)軌道ベース最適化スモークテスト(research_plan.md フェーズ2の先取り)。

synthetic_recovery_test.py で検証した「攻撃者1人+守備者N人」の最小構成を、
実データ(match J03WPY)の実際のプレッシャー局面に適用する。

- ボール保持者(攻撃者役)の特定は check_distance_range.py と同じ近似(最近傍選手)を利用し、
  同一選手が保持し続ける連続区間(run)から、守備者が最も寄せてきている区間を選ぶ。
- 各エンティティの「意図した速度」v0*e(t) は、観測軌道を長い窓(2秒)でSavitzky-Golay
  平滑化したものを既知の入力として扱う(3.2節のe_i(t), v0_i(t)の簡易近似)。
- 相互作用項なし(A1=A2=0、intentに従うだけ)をベースラインとし、
  フィッティングでどれだけ改善するかを比較することで、SFMの説明力を評価する。
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from torchdiffeq import odeint

from sfm_press.data import build_snapshot

plt.rcParams["font.family"] = "Hiragino Sans"
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

MATCH_ID = "J03WPY"
N_DEF = 3
R0 = 0.3
FPS = 25
DT = 1.0 / FPS
INTENT_WINDOW = 51  # 2.04秒、長期的な「意図した方向」を表す平滑化窓
INTENT_POLYORDER = 2


def find_best_window(df: pd.DataFrame, min_len: int = 150) -> dict:
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
        period=("period_id", "first"),
    )
    runs = runs[runs["n"] >= min_len].sort_values("n", ascending=False)

    best = None
    for _, run in runs.head(15).iterrows():
        start, end, cid = int(run["start"]), int(run["end"]), run["carrier_id"]
        window = df[(df["frame_id"] >= start) & (df["frame_id"] <= end)]
        carrier_team = window[window["entity_id"] == cid]["team_id"].iloc[0]
        opp = window[(window["entity_id"] != "ball") & (window["team_id"] != carrier_team)]
        c_pos = window[window["entity_id"] == cid][["frame_id", "x", "y"]].rename(
            columns={"x": "cx", "y": "cy"}
        )
        opp = opp.merge(c_pos, on="frame_id")
        opp["dist"] = np.hypot(opp["x"] - opp["cx"], opp["y"] - opp["cy"])
        by_entity = opp.groupby("entity_id")["dist"]
        avg_dist = by_entity.mean().sort_values()
        top3 = avg_dist.head(N_DEF).index.tolist()
        if len(top3) < N_DEF:
            continue
        nearest = top3[0]
        d = opp[opp["entity_id"] == nearest].sort_values("frame_id")["dist"]
        closing = d.iloc[0] - d.iloc[-1]  # 正なら距離が縮まっている(プレッシャーがかかっている)
        score = closing
        if best is None or score > best["score"]:
            best = dict(start=start, end=end, carrier_id=cid, defender_ids=top3, score=score)
    return best


def intent_signal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """長い窓のSG平滑化から「意図した速度」(vx0, vy0)を推定する。"""
    vx0 = savgol_filter(x, INTENT_WINDOW, INTENT_POLYORDER, deriv=1, delta=DT)
    vy0 = savgol_filter(y, INTENT_WINDOW, INTENT_POLYORDER, deriv=1, delta=DT)
    return vx0, vy0


def social_force_ode(A1, B1, A2, B2, tau_att, tau_def, intent, t_eval):
    """intent: (N_STEPS, 1+Ndef, 2) の意図速度テンソル。tをframe indexに丸めて参照する。"""
    n_steps = intent.shape[0]
    t0, t1 = t_eval[0].item(), t_eval[-1].item()

    def f(t, state):
        idx = int(round((t.item() - t0) / (t1 - t0) * (n_steps - 1)))
        idx = max(0, min(n_steps - 1, idx))
        v0 = intent[idx]  # (1+Ndef, 2)

        pos = state[..., :2]
        vel = state[..., 2:4]
        att_pos, def_pos = pos[0:1, :], pos[1:, :]
        att_vel, def_vel = vel[0, :], vel[1:, :]

        drive_att = (v0[0] - att_vel) / tau_att
        drive_def = (v0[1:] - def_vel) / tau_def

        diff1 = att_pos - def_pos
        dist1 = diff1.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        f_att_from_def = (A1 * torch.exp((R0 - dist1) / B1) * diff1 / dist1).sum(dim=0)

        diff2 = def_pos - att_pos
        dist2 = diff2.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        f_def_from_att = A2 * torch.exp((R0 - dist2) / B2) * diff2 / dist2

        accel_att = drive_att + f_att_from_def
        accel_def = drive_def + f_def_from_att

        d_att = torch.cat([att_vel, accel_att]).unsqueeze(0)
        d_def = torch.cat([def_vel, accel_def], dim=-1)
        return torch.cat([d_att, d_def], dim=0)

    return f


def main():
    df = build_snapshot(MATCH_ID)
    best = find_best_window(df)
    print("selected window:", best)

    entities = [best["carrier_id"]] + best["defender_ids"]
    window = df[(df["frame_id"] >= best["start"]) & (df["frame_id"] <= best["end"])]
    n_steps = window["frame_id"].nunique()
    t_eval = torch.linspace(0, (n_steps - 1) * DT, n_steps)

    pos_obs = np.zeros((n_steps, len(entities), 2))
    vel0 = np.zeros((len(entities), 2))
    intent = np.zeros((n_steps, len(entities), 2))

    for i, eid in enumerate(entities):
        g = window[window["entity_id"] == eid].sort_values("frame_id")
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        pos_obs[:, i, 0], pos_obs[:, i, 1] = x, y
        vel0[i] = g["vx"].iloc[0], g["vy"].iloc[0]
        vx0, vy0 = intent_signal(x, y)
        intent[:, i, 0], intent[:, i, 1] = vx0, vy0

    pos_obs_t = torch.tensor(pos_obs)
    intent_t = torch.tensor(intent)

    y0 = torch.zeros(1 + N_DEF, 4)
    y0[:, :2] = pos_obs_t[0]
    y0[:, 2:4] = torch.tensor(vel0)

    def simulate(params):
        f = social_force_ode(**params, intent=intent_t, t_eval=t_eval)
        traj = odeint(f, y0, t_eval, method="rk4", options={"step_size": DT / 2})
        return traj

    # --- ベースライン(相互作用なし、intentに従うだけ) ---
    with torch.no_grad():
        baseline_params = dict(
            A1=torch.tensor(0.0), B1=torch.tensor(1.0),
            A2=torch.tensor(0.0), B2=torch.tensor(1.0),
            tau_att=torch.tensor(0.5), tau_def=torch.tensor(0.5),
        )
        baseline_traj = simulate(baseline_params)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    print(f"baseline (no interaction) MSE: {baseline_loss:.4f}  RMSE: {np.sqrt(baseline_loss):.3f} m")

    # --- フィッティング ---
    raw = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
        "log_tau_att": torch.tensor(np.log(0.5), requires_grad=True),
        "log_tau_def": torch.tensor(np.log(0.5), requires_grad=True),
    }

    def get_params():
        return dict(
            A1=raw["A1"], B1=torch.exp(raw["log_B1"]),
            A2=raw["A2"], B2=torch.exp(raw["log_B2"]),
            tau_att=torch.exp(raw["log_tau_att"]), tau_def=torch.exp(raw["log_tau_def"]),
        )

    optimizer = torch.optim.Adam(raw.values(), lr=0.05)
    n_opt_steps = 400
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
        "window": {"start": best["start"], "end": best["end"], "carrier_id": best["carrier_id"],
                   "defender_ids": best["defender_ids"], "n_steps": n_steps},
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {k: v.item() for k, v in final_params.items()},
        "history": history,
        "entities": entities,
        "pos_obs": pos_obs.tolist(),
        "pos_pred_baseline": baseline_traj[..., :2].numpy().tolist(),
        "pos_pred_fitted": final_traj[..., :2].detach().numpy().tolist(),
    }
    with open("scripts/real_data_fit_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/real_data_fit_result.json")


if __name__ == "__main__":
    main()
