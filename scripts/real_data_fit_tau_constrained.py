"""real_data_fit_test.py のτ縮退を防ぐため、τに物理的に妥当な範囲の制約を加えた再実験。

同じ区間・同じ意図速度信号を使い、τだけをsigmoidで[TAU_MIN, TAU_MAX]の範囲に
ハードに制約する。この制約下でも軌道再現精度がベースラインに対して十分改善するかどうかで、
「A, Bが実際に意味のある仕事をしているか」「v0*e(t)の設計自体を見直すべきか」を切り分ける。
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from scripts.real_data_fit_test import (
    DT,
    MATCH_ID,
    N_DEF,
    build_snapshot,
    find_best_window,
    intent_signal,
    social_force_ode,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_MIN = 0.3
TAU_MAX = 1.2


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

    # sigmoid(raw)=0.5 (tau=0.75s) からスタート
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
    print(f"(tau constrained to [{TAU_MIN}, {TAU_MAX}])")

    out = {
        "match_id": MATCH_ID,
        "tau_min": TAU_MIN,
        "tau_max": TAU_MAX,
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
    with open("scripts/real_data_fit_tau_constrained_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/real_data_fit_tau_constrained_result.json")


if __name__ == "__main__":
    main()
