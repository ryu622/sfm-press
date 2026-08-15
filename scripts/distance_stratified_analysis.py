"""近接局面に絞った改善率の再集計(real_data_fit_test.md 追試8)。

追試7(tau_fixed_ab_only.py)の全フレーム平均の改善率(最大7.3%)は、
守備者から離れていて相互作用力がほぼゼロに近い局面(指数関数的な力のため)を
大量に含んでいる可能性がある。この追試では、tau_fixed_ab_only.pyで
既に推定済みのA,Bを再利用し、フレームごとの誤差改善を「その瞬間の
最近傍守備者との距離」でビン分けして、近接局面(プレッシャーが強いはずの
局面)に絞ったときに改善率がどう変わるかを確認する。

- 距離に依らず改善率が低いままなら、相互作用項の定式化自体に問題がある
  可能性が強まる(仮説①)。
- 近接局面で明確に改善するなら、全体平均が薄めていただけという解釈になる
  (仮説②寄りだが、定式化そのものの評価はまだ保留)。
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

DIST_BINS = [0, 2, 4, 6, 10, 15, 25, 100]  # メートル


def simulate(pos_obs_t, vel0, intent_t, t_eval, n_ep, A1, B1, A2, B2, tau_att, tau_def):
    y0 = torch.zeros(n_ep, 1 + N_DEF, 4)
    y0[:, :, :2] = pos_obs_t[0]
    y0[:, :, 2:4] = vel0
    f = social_force_ode_v3(A1=A1, B1=B1, A2=A2, B2=B2, tau_att=tau_att, tau_def=tau_def,
                             intent=intent_t, t_eval=t_eval)
    traj = odeint(f, y0, t_eval, method="rk4", options={"step_size": DT / 2})
    return traj


def main():
    with open("scripts/tau_fixed_ab_only_result.json") as f:
        fixed_tau_data = json.load(f)

    df = build_snapshot(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    pos_obs_t, vel0, intent_t = build_batch(df, windows, EMA_HALFLIFE_SEC)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(windows)

    # 観測軌道における「攻撃者<->最近傍守備者」距離 (n_steps, n_ep)
    att_pos_obs = pos_obs_t[:, :, 0, :]         # (T, E, 2)
    def_pos_obs = pos_obs_t[:, :, 1:, :]        # (T, E, Ndef, 2)
    dist_obs = (att_pos_obs.unsqueeze(2) - def_pos_obs).norm(dim=-1)  # (T, E, Ndef)
    nearest_dist_obs = dist_obs.min(dim=-1).values.numpy()  # (T, E)

    bin_edges = np.array(DIST_BINS)
    n_bins = len(bin_edges) - 1

    all_results = {}
    for r in fixed_tau_data["results"]:
        tau_fixed = r["tau_fixed"]
        p = r["final_params"]
        tau_att = torch.tensor(tau_fixed)
        tau_def = torch.tensor(tau_fixed)

        with torch.no_grad():
            baseline_traj = simulate(pos_obs_t, vel0, intent_t, t_eval, n_ep,
                                      torch.tensor(0.0), torch.tensor(1.0),
                                      torch.tensor(0.0), torch.tensor(1.0), tau_att, tau_def)
            fitted_traj = simulate(pos_obs_t, vel0, intent_t, t_eval, n_ep,
                                    torch.tensor(p["A1"]), torch.tensor(p["B1"]),
                                    torch.tensor(p["A2"]), torch.tensor(p["B2"]), tau_att, tau_def)

        sq_err_baseline = ((baseline_traj[..., :2] - pos_obs_t) ** 2).sum(dim=-1)  # (T, E, 1+Ndef)
        sq_err_fitted = ((fitted_traj[..., :2] - pos_obs_t) ** 2).sum(dim=-1)

        # 攻撃者のみの誤差に着目(相互作用項の主戦場)
        att_err_baseline = sq_err_baseline[:, :, 0].numpy()  # (T, E)
        att_err_fitted = sq_err_fitted[:, :, 0].numpy()

        bin_stats = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (nearest_dist_obs >= lo) & (nearest_dist_obs < hi)
            n_frames = int(mask.sum())
            if n_frames == 0:
                bin_stats.append(dict(lo=float(lo), hi=float(hi), n_frames=0,
                                       baseline_mse=None, fitted_mse=None, improvement_pct=None))
                continue
            base_mse = float(att_err_baseline[mask].mean())
            fit_mse = float(att_err_fitted[mask].mean())
            improve = (1 - fit_mse / base_mse) * 100 if base_mse > 0 else None
            bin_stats.append(dict(lo=float(lo), hi=float(hi), n_frames=n_frames,
                                   baseline_mse=base_mse, fitted_mse=fit_mse, improvement_pct=improve))

        all_results[str(tau_fixed)] = bin_stats

        print(f"\n=== tau_fixed={tau_fixed}s (overall improvement in original test: "
              f"{r['improvement_pct']:.1f}%) ===")
        print(f"{'dist range':>14s}  {'n_frames':>9s}  {'base_RMSE':>10s}  {'fit_RMSE':>9s}  {'improve%':>9s}")
        for s in bin_stats:
            if s["n_frames"] == 0:
                print(f"{s['lo']:>5.0f}-{s['hi']:<7.0f}m  {'0':>9s}")
                continue
            print(f"{s['lo']:>5.0f}-{s['hi']:<7.0f}m  {s['n_frames']:>9d}  "
                  f"{s['baseline_mse']**0.5:>10.3f}  {s['fitted_mse']**0.5:>9.3f}  "
                  f"{s['improvement_pct']:>9.1f}")

    with open("scripts/distance_stratified_result.json", "w") as f:
        json.dump({"dist_bins": DIST_BINS, "by_tau": all_results}, f, indent=2)
    print("\nsaved: scripts/distance_stratified_result.json")


if __name__ == "__main__":
    main()
