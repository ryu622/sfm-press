"""動的ポテンシャル駆動力(v4)でのτ固定診断テスト、フェア版(real_data_fit_test.md 追試11)。

追試10のτ固定テスト(tau_fixed_v4_sweep.py)は、A,Bと同時にv0_att,v0_defも
自由にしていたため、「駆動力側のチューニング改善」と「相互作用項の改善」が
混ざってしまい、追試7(EMA版、A,Bのみ自由)とフェアに比較できていなかった。

この追試では、各τ固定値について2段階で最適化する:
  段階1: A1=B1=A2=B2=0に固定し、v0_att, v0_defだけを最適化
         (この駆動力設計だけで到達できる最良のベースラインを確定)
  段階2: 段階1で得たv0を固定し、A1,B1,A2,B2だけを最適化
         (相互作用項だけの純粋な追加改善を測る)

これで追試7と全く同じ土俵(「駆動力側は完全に固定し、A,Bだけの追加改善を見る」)
で比較できる。
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from scripts.real_data_pooled_fit import find_pooled_windows, N_WINDOWS, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, N_DEF, R0, FPS, DT, build_snapshot
from scripts.real_data_pooled_fit_v4 import build_batch, social_force_ode_v4
from sfm_press.data import determine_attacking_goal_x

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_FIXED_LEVELS = [0.1, 0.3, 0.5, 0.8, 1.2]
N_OPT_STEPS = 1000


def simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, A1, B1, A2, B2, tau_att, tau_def, v0_att, v0_def):
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0
    f = social_force_ode_v4(A1=A1, B1=B1, A2=A2, B2=B2, tau_att=tau_att, tau_def=tau_def,
                             v0_att=v0_att, v0_def=v0_def, goal_pos=goal_pos_t)
    traj = odeint(f, y0, t_eval, method="rk4", options={"step_size": DT / 2})
    return traj


def run_two_stage(df, windows, attack_goal, tau_fixed: float, n_opt_steps: int, verbose: bool = True) -> dict:
    pos_obs_t, vel0, goal_pos_t = build_batch(df, windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(windows)

    tau_att = torch.tensor(tau_fixed)
    tau_def = torch.tensor(tau_fixed)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    # --- 段階1: A=B=0固定、v0だけ最適化(この駆動力設計の最良ベースライン) ---
    raw_v0 = {
        "log_v0_att": torch.tensor(np.log(4.0), requires_grad=True),
        "log_v0_def": torch.tensor(np.log(2.0), requires_grad=True),
    }
    opt1 = torch.optim.Adam(raw_v0.values(), lr=0.05)
    for step in range(n_opt_steps):
        opt1.zero_grad()
        v0_att = torch.exp(raw_v0["log_v0_att"])
        v0_def = torch.exp(raw_v0["log_v0_def"])
        traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, zero, one, zero, one,
                         tau_att, tau_def, v0_att, v0_def)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        opt1.step()
        if verbose and (step % 100 == 0 or step == n_opt_steps - 1):
            print(f"[tau={tau_fixed}s stage1] step {step:4d}  loss={loss.item():.4f}  "
                  f"RMSE={np.sqrt(loss.item()):.3f}m  v0_att={v0_att.item():.3f} v0_def={v0_def.item():.3f}")

    v0_att_fixed = torch.exp(raw_v0["log_v0_att"]).detach()
    v0_def_fixed = torch.exp(raw_v0["log_v0_def"]).detach()
    with torch.no_grad():
        baseline_traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, zero, one, zero, one,
                                  tau_att, tau_def, v0_att_fixed, v0_def_fixed)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    if verbose:
        print(f"[tau={tau_fixed}s] drive-only best baseline: MSE={baseline_loss:.4f}  "
              f"RMSE={np.sqrt(baseline_loss):.3f}m  (v0_att={v0_att_fixed.item():.3f}, "
              f"v0_def={v0_def_fixed.item():.3f})")

    # --- 段階2: v0固定、A,Bだけ最適化(相互作用項単独の追加改善) ---
    raw_ab = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
    }
    opt2 = torch.optim.Adam(raw_ab.values(), lr=0.05)
    history = {"step": [], "loss": [], "A1": [], "B1": [], "A2": [], "B2": []}
    for step in range(n_opt_steps):
        opt2.zero_grad()
        A1, B1 = raw_ab["A1"], torch.exp(raw_ab["log_B1"])
        A2, B2 = raw_ab["A2"], torch.exp(raw_ab["log_B2"])
        traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, A1, B1, A2, B2,
                         tau_att, tau_def, v0_att_fixed, v0_def_fixed)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        opt2.step()

        history["step"].append(step)
        history["loss"].append(loss.item())
        history["A1"].append(A1.item())
        history["B1"].append(B1.item())
        history["A2"].append(A2.item())
        history["B2"].append(B2.item())

        if verbose and (step % 100 == 0 or step == n_opt_steps - 1):
            print(f"[tau={tau_fixed}s stage2] step {step:4d}  loss={loss.item():.4f}  "
                  f"RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={A1.item():+.3f} B1={B1.item():.3f} A2={A2.item():+.3f} B2={B2.item():.3f}")

    final_loss = history["loss"][-1]
    improvement = (1 - final_loss / baseline_loss) * 100
    if verbose:
        print(f"[tau={tau_fixed}s] final: RMSE={np.sqrt(final_loss):.3f}m  improvement={improvement:.1f}%\n")

    return {
        "tau_fixed": tau_fixed,
        "v0_att_fixed": v0_att_fixed.item(),
        "v0_def_fixed": v0_def_fixed.item(),
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {"A1": history["A1"][-1], "B1": history["B1"][-1],
                          "A2": history["A2"][-1], "B2": history["B2"][-1]},
        "history": history,
    }


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    results = []
    for tau_fixed in TAU_FIXED_LEVELS:
        print(f"\n########## tau_fixed = {tau_fixed}s ##########")
        out = run_two_stage(df, windows, attack_goal, tau_fixed, N_OPT_STEPS)
        results.append(out)

    print("\n=== summary: fair A,B-only improvement (v4 drive, v0 pre-fit and frozen) ===")
    header = f"{'tau[s]':>7s}  {'base_RMSE':>10s}  {'fit_RMSE':>9s}  {'improve%':>9s}  " \
             f"{'v0_att':>7s}  {'v0_def':>7s}"
    print(header)
    for r in results:
        print(f"{r['tau_fixed']:>7.2f}  {r['baseline_mse']**0.5:>10.3f}  "
              f"{r['final_mse']**0.5:>9.3f}  {r['improvement_pct']:>9.1f}  "
              f"{r['v0_att_fixed']:>7.3f}  {r['v0_def_fixed']:>7.3f}")

    with open("scripts/tau_fixed_v4_fair_result.json", "w") as f:
        json.dump({"tau_fixed_levels": TAU_FIXED_LEVELS, "results": results}, f)
    print("\nsaved: scripts/tau_fixed_v4_fair_result.json")


if __name__ == "__main__":
    main()
