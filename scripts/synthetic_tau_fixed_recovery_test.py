"""合成データでτを真値に固定し、A,Bだけの識別可能性を確認する(real_data_fit_test.md 追試9)。

synthetic_recovery_test.py はτも自由パラメータとして推定させており、
実データの追試7(tau_fixed_ab_only.py, τを固定してA,Bのみ推定)とは
条件が異なっていた。この追試ではτを真値に固定した上でA1,B1,A2,B2のみを
推定し、実データと全く同じ縛りの下でも合成データなら復元できるかを確認する。

- 復元に失敗する(誤差が大きい/ベースラインからの改善が乏しい)なら、
  τを固定するという制約そのものが識別可能性を損なっている可能性が高く、
  実データでの追試7の結果(改善率が乏しい)はモデル定式化・識別可能性
  そのものの問題である可能性が強まる(仮説①)。
- 復元に成功するなら、合成データ(ノイズなし・モデル完全一致)では
  τ固定でも問題なく、実データでの伸び悩みはデータの粗さや駆動力信号の
  設計に起因する可能性が高まる(仮説②)。
"""

import json

import numpy as np
import torch
from torchdiffeq import odeint

from scripts.synthetic_recovery_test import (
    N_DEF, N_EPISODES, T_EVAL, TRUE, R0, make_initial_states, social_force_ode,
)

torch.set_default_dtype(torch.float64)


def simulate(A1, B1, A2, B2, tau_att, tau_def, y0):
    f = social_force_ode(A1=A1, B1=B1, A2=A2, B2=B2, tau_att=tau_att, tau_def=tau_def)
    traj = odeint(f, y0, T_EVAL, method="rk4", options={"step_size": 0.02})
    return traj


def run_experiment(seed: int = 0, n_steps: int = 300, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    y0 = make_initial_states(N_EPISODES, N_DEF)

    tau_att_true = torch.tensor(TRUE["tau_att"])
    tau_def_true = torch.tensor(TRUE["tau_def"])

    with torch.no_grad():
        traj_true = simulate(torch.tensor(TRUE["A1"]), torch.tensor(TRUE["B1"]),
                              torch.tensor(TRUE["A2"]), torch.tensor(TRUE["B2"]),
                              tau_att_true, tau_def_true, y0)
    pos_obs = traj_true[..., :2].detach()

    with torch.no_grad():
        baseline_traj = simulate(torch.tensor(0.0), torch.tensor(1.0),
                                  torch.tensor(0.0), torch.tensor(1.0),
                                  tau_att_true, tau_def_true, y0)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs) ** 2).mean().item()
    if verbose:
        print(f"baseline (no interaction, tau=true) MSE: {baseline_loss:.4f}  "
              f"RMSE: {np.sqrt(baseline_loss):.3f} m")

    # A,Bのみ推定対象(意図的に真値からずらして初期化、符号も逆から)
    raw = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(2.0), requires_grad=True),
        "A2": torch.tensor(1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(2.0), requires_grad=True),
    }

    def get_params():
        return dict(A1=raw["A1"], B1=torch.exp(raw["log_B1"]),
                    A2=raw["A2"], B2=torch.exp(raw["log_B2"]))

    optimizer = torch.optim.Adam(raw.values(), lr=0.08)
    history = {"step": [], "loss": [], "A1": [], "B1": [], "A2": [], "B2": []}

    for step in range(n_steps):
        optimizer.zero_grad()
        p = get_params()
        traj_pred = simulate(p["A1"], p["B1"], p["A2"], p["B2"], tau_att_true, tau_def_true, y0)
        loss = ((traj_pred[..., :2] - pos_obs) ** 2).mean()
        loss.backward()
        optimizer.step()

        pv = {k: v.item() for k, v in p.items()}
        history["step"].append(step)
        history["loss"].append(loss.item())
        for k in ["A1", "B1", "A2", "B2"]:
            history[k].append(pv[k])

        if verbose and (step % 20 == 0 or step == n_steps - 1):
            print(f"step {step:4d}  loss={loss.item():.5f}  RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={pv['A1']:+.3f} B1={pv['B1']:.3f} A2={pv['A2']:+.3f} B2={pv['B2']:.3f}")

    final = get_params()
    with torch.no_grad():
        final_traj = simulate(final["A1"], final["B1"], final["A2"], final["B2"],
                               tau_att_true, tau_def_true, y0)
    final_loss = ((final_traj[..., :2] - pos_obs) ** 2).mean().item()
    improvement = (1 - final_loss / baseline_loss) * 100

    summary = {}
    for k in ["A1", "B1", "A2", "B2"]:
        true_v = TRUE[k]
        est_v = final[k].item()
        rel_err = abs(est_v - true_v) / (abs(true_v) + 1e-8) * 100
        sign_ok = np.sign(true_v) == np.sign(est_v)
        summary[k] = {"true": true_v, "recovered": est_v, "rel_err_pct": rel_err, "sign_ok": bool(sign_ok)}
        if verbose:
            print(f"  {k:10s}  true={true_v:+.3f}  recovered={est_v:+.3f}  rel_err={rel_err:5.1f}%  "
                  f"{'OK' if sign_ok else 'SIGN FLIPPED'}")

    if verbose:
        print(f"\nfinal fitted MSE: {final_loss:.4f}  RMSE: {np.sqrt(final_loss):.3f} m  "
              f"improvement: {improvement:.1f}%")

    return {
        "seed": seed,
        "n_episodes": N_EPISODES,
        "n_def": N_DEF,
        "true_params": TRUE,
        "tau_att_fixed": TRUE["tau_att"],
        "tau_def_fixed": TRUE["tau_def"],
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "history": history,
        "summary": summary,
    }


def main():
    out = run_experiment()
    with open("scripts/synthetic_tau_fixed_recovery_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved: scripts/synthetic_tau_fixed_recovery_result.json")


if __name__ == "__main__":
    main()
