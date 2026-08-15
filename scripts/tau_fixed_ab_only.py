"""τを固定し、A,Bだけの説明力を直接テストする(real_data_fit_test.md 追試7)。

追試6までで、τを自由パラメータにすると常に「その時点で許される最小値」に
張り付き続け、駆動力への丸投げが起きていることが分かった。この追試では
τを最適化対象から外して固定値にし、相互作用項A1,B1,A2,B2だけで
ベースライン(相互作用なし)からどれだけ改善できるかを直接確認する。

これは駆動力設計(EMA半減期など)の選択とは独立に、
「SFMの相互作用項に守備プレッシャーを説明する実質的な力があるか」への
直接的な診断になる。
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from scripts.real_data_pooled_fit import find_pooled_windows, N_WINDOWS, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, N_DEF, R0, FPS, DT, build_snapshot
from scripts.real_data_pooled_fit_v3 import build_batch, social_force_ode_v3, EMA_HALFLIFE_SEC

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_FIXED_LEVELS = [0.1, 0.3, 0.5, 0.8, 1.2]
N_OPT_STEPS = 1200


def run_fixed_tau_experiment(df, windows: list[dict], halflife_sec: float, tau_fixed: float,
                              n_opt_steps: int = 1200, verbose: bool = True) -> dict:
    pos_obs_t, vel0, intent_t = build_batch(df, windows, halflife_sec)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)

    n_ep = len(windows)
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0

    tau_att = torch.tensor(tau_fixed)
    tau_def = torch.tensor(tau_fixed)

    def simulate(A1, B1, A2, B2):
        f = social_force_ode_v3(A1=A1, B1=B1, A2=A2, B2=B2, tau_att=tau_att, tau_def=tau_def,
                                 intent=intent_t, t_eval=t_eval)
        traj = odeint(f, y0, t_eval, method="rk4", options={"step_size": DT / 2})
        return traj

    with torch.no_grad():
        baseline_traj = simulate(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0))
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    if verbose:
        print(f"[tau_fixed={tau_fixed}s] baseline (no interaction, same tau) MSE: {baseline_loss:.4f}  "
              f"RMSE: {np.sqrt(baseline_loss):.3f} m")

    raw = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
    }

    def get_params():
        return dict(A1=raw["A1"], B1=torch.exp(raw["log_B1"]),
                    A2=raw["A2"], B2=torch.exp(raw["log_B2"]))

    optimizer = torch.optim.Adam(raw.values(), lr=0.05)
    history = {"step": [], "loss": [], "A1": [], "B1": [], "A2": [], "B2": []}

    for step in range(n_opt_steps):
        optimizer.zero_grad()
        p = get_params()
        traj = simulate(p["A1"], p["B1"], p["A2"], p["B2"])
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        optimizer.step()

        pv = {k: v.item() for k, v in p.items()}
        history["step"].append(step)
        history["loss"].append(loss.item())
        for k in ["A1", "B1", "A2", "B2"]:
            history[k].append(pv[k])

        if verbose and (step % 40 == 0 or step == n_opt_steps - 1):
            print(f"[tau_fixed={tau_fixed}s] step {step:4d}  loss={loss.item():.4f}  "
                  f"RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={pv['A1']:+.3f} B1={pv['B1']:.3f} A2={pv['A2']:+.3f} B2={pv['B2']:.3f}")

    final_p = get_params()
    with torch.no_grad():
        final_traj = simulate(final_p["A1"], final_p["B1"], final_p["A2"], final_p["B2"])
    final_loss = ((final_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    improvement = (1 - final_loss / baseline_loss) * 100
    if verbose:
        print(f"[tau_fixed={tau_fixed}s] final fitted MSE: {final_loss:.4f}  "
              f"RMSE: {np.sqrt(final_loss):.3f} m  improvement: {improvement:.1f}%")

    return {
        "match_id": MATCH_ID,
        "n_windows": n_ep,
        "fixed_len": FIXED_LEN,
        "ema_halflife_sec": halflife_sec,
        "tau_fixed": tau_fixed,
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "final_params": {k: v.item() for k, v in final_p.items()},
        "history": history,
    }


def main():
    df = build_snapshot(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    results = []
    for tau_fixed in TAU_FIXED_LEVELS:
        out = run_fixed_tau_experiment(df, windows, EMA_HALFLIFE_SEC, tau_fixed, n_opt_steps=N_OPT_STEPS)
        results.append(out)
        print()

    print("=== summary across fixed tau values ===")
    header = f"{'tau_fixed[s]':>12s}  {'baseline_RMSE':>13s}  {'fitted_RMSE':>11s}  {'improve%':>9s}  "
    header += "  ".join(f"{k:>10s}" for k in ["A1", "B1", "A2", "B2"])
    print(header)
    for r in results:
        p = r["final_params"]
        row = (f"{r['tau_fixed']:>12.2f}  {r['baseline_mse']**0.5:>13.3f}  "
               f"{r['final_mse']**0.5:>11.3f}  {r['improvement_pct']:>9.1f}  ")
        row += "  ".join(f"{p[k]:>10.3f}" for k in ["A1", "B1", "A2", "B2"])
        print(row)

    with open("scripts/tau_fixed_ab_only_result.json", "w") as f:
        json.dump({"tau_fixed_levels": TAU_FIXED_LEVELS, "results": results}, f)
    print("\nsaved: scripts/tau_fixed_ab_only_result.json")


if __name__ == "__main__":
    main()
