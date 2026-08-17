"""5分割交差検証で、全40局面をheld-outとして評価し改善率の推定精度を上げる(追試16)。

追試15で、train30/test10のランダム分割を5シード試したところ、test改善率が
1.0〜13.4%(std=4.1%)と大きくばらつくことが分かった。これはtestサイズが
10局面と小さいことによる統計的な粗さが原因と考えられる。この追試では
40局面を5分割(各8局面)し、毎回どれか1つを held-out にして残り32局面で
学習する交差検証を行うことで、全40局面が1回ずつheld-outとして評価される
ようにし、実効的なtestサンプル数を10→40に増やして改善率の推定精度を上げる。
"""

import json

import numpy as np
import torch

from scripts.real_data_pooled_fit import find_pooled_windows, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, build_snapshot, DT
from scripts.real_data_pooled_fit_v4 import build_batch
from scripts.tau_fixed_v4_fair_comparison import simulate
from scripts.held_out_validation import eval_on
from sfm_press.data import determine_attacking_goal_x

torch.set_default_dtype(torch.float64)

TAU_FIXED = 1.2
N_WINDOWS = 40
K = 5
N_STEPS_STAGE1 = 1000
N_STEPS_STAGE2 = 3000
SPLIT_SEED = 0


def run_fold(df, train_windows, test_windows, attack_goal, fold_id):
    torch.manual_seed(fold_id)
    pos_obs_t, vel0, goal_pos_t = build_batch(df, train_windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(train_windows)
    tau = torch.tensor(TAU_FIXED)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    raw_v0 = {
        "log_v0_att": torch.tensor(np.log(4.0), requires_grad=True),
        "log_v0_def": torch.tensor(np.log(2.0), requires_grad=True),
    }
    opt1 = torch.optim.Adam(raw_v0.values(), lr=0.05)
    for step in range(N_STEPS_STAGE1):
        opt1.zero_grad()
        v0_att = torch.exp(raw_v0["log_v0_att"])
        v0_def = torch.exp(raw_v0["log_v0_def"])
        traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, zero, one, zero, one,
                         tau, tau, v0_att, v0_def)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        opt1.step()
    v0_att_fixed = torch.exp(raw_v0["log_v0_att"]).detach()
    v0_def_fixed = torch.exp(raw_v0["log_v0_def"]).detach()

    raw_ab = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
    }
    opt2 = torch.optim.Adam(raw_ab.values(), lr=0.05)
    for step in range(N_STEPS_STAGE2):
        opt2.zero_grad()
        A1, B1 = raw_ab["A1"], torch.exp(raw_ab["log_B1"])
        A2, B2 = raw_ab["A2"], torch.exp(raw_ab["log_B2"])
        traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, A1, B1, A2, B2,
                         tau, tau, v0_att_fixed, v0_def_fixed)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        opt2.step()

    A1_final, B1_final = raw_ab["A1"].detach(), torch.exp(raw_ab["log_B1"]).detach()
    A2_final, B2_final = raw_ab["A2"].detach(), torch.exp(raw_ab["log_B2"]).detach()

    test_eval = eval_on(test_windows, attack_goal, df, tau, A1_final, B1_final, A2_final, B2_final,
                         v0_att_fixed, v0_def_fixed)

    print(f"[fold={fold_id}] n_train={len(train_windows)} n_test={len(test_windows)}  "
          f"test_baseline_RMSE={test_eval['baseline_mse']**0.5:.3f}m  "
          f"test_fitted_RMSE={test_eval['fitted_mse']**0.5:.3f}m  "
          f"test_improve={test_eval['improvement_pct']:.1f}%  "
          f"A1={A1_final.item():+.3f} B1={B1_final.item():.3f} "
          f"A2={A2_final.item():+.3f} B2={B2_final.item():.3f}")

    return {
        "fold": fold_id,
        "n_train": len(train_windows),
        "n_test": len(test_windows),
        "test_baseline_mse": test_eval["baseline_mse"],
        "test_fitted_mse": test_eval["fitted_mse"],
        "test_improve": test_eval["improvement_pct"],
        "final_params": {"A1": A1_final.item(), "B1": B1_final.item(),
                          "A2": A2_final.item(), "B2": B2_final.item()},
    }


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    all_windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"total windows: {len(all_windows)}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(all_windows))
    fold_size = len(all_windows) // K
    folds = [perm[i * fold_size:(i + 1) * fold_size] for i in range(K)]
    print(f"K={K}  fold sizes: {[len(f) for f in folds]}\n")

    results = []
    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        train_windows = [all_windows[i] for i in train_idx]
        test_windows = [all_windows[i] for i in test_idx]
        results.append(run_fold(df, train_windows, test_windows, attack_goal, k))

    # プールされたheld-out評価(全40局面の held-out 予測誤差を合算)
    total_baseline = np.mean([r["test_baseline_mse"] for r in results])
    total_fitted = np.mean([r["test_fitted_mse"] for r in results])
    pooled_improve = (1 - total_fitted / total_baseline) * 100

    fold_improves = [r["test_improve"] for r in results]
    print(f"\nper-fold test improve: {[f'{v:.1f}%' for v in fold_improves]}")
    print(f"per-fold mean={np.mean(fold_improves):.1f}%  std={np.std(fold_improves):.1f}%")
    print(f"pooled (全40局面held-out, フォールドサイズが等しいため単純平均と一致): "
          f"baseline_RMSE={total_baseline**0.5:.3f}m  fitted_RMSE={total_fitted**0.5:.3f}m  "
          f"improvement={pooled_improve:.1f}%")

    out = {
        "k": K,
        "n_windows": N_WINDOWS,
        "tau_fixed": TAU_FIXED,
        "split_seed": SPLIT_SEED,
        "results": results,
        "pooled_baseline_mse": total_baseline,
        "pooled_fitted_mse": total_fitted,
        "pooled_improvement_pct": pooled_improve,
        "fold_improve_mean": float(np.mean(fold_improves)),
        "fold_improve_std": float(np.std(fold_improves)),
    }
    with open("scripts/kfold_cv_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/kfold_cv_result.json")


if __name__ == "__main__":
    main()
