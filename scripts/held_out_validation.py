"""相互作用項の改善率がheld-out局面でも残るかを検証する(real_data_fit_test.md 追試14)。

これまでの改善率(7〜19%、追試13では9.4%)は全てin-sample、つまりA,Bのフィットに
使った局面そのもので評価していた。この追試では、追試13と同じ40局面をtrain(30)/
test(10)にランダム分割し、trainだけでv0とA,Bを最適化(τ=1.2秒固定、追試13と同じ
手続き)、その結果をtestに一度も学習に使わず適用して、held-out改善率を測る。

train改善率とtest改善率がほぼ同水準なら、相互作用項の効果は特定の局面への
過学習ではなく汎化する本物の効果と言える。testで大きく潰れるなら、
これまでの改善率は局面選択(closing distanceが大きい局面を上位40件選ぶという
選定基準自体)への過学習だった可能性が高く、路線の見直しが必要になる。
"""

import json

import numpy as np
import torch

from scripts.real_data_pooled_fit import find_pooled_windows, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, build_snapshot, DT
from scripts.real_data_pooled_fit_v4 import build_batch
from scripts.tau_fixed_v4_fair_comparison import simulate
from sfm_press.data import determine_attacking_goal_x

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_FIXED = 1.2
N_WINDOWS = 40
N_TEST = 10
N_STEPS_STAGE1 = 1000
N_STEPS_STAGE2 = 3000
SPLIT_SEED = 0


def eval_on(windows, attack_goal, df, tau, A1, B1, A2, B2, v0_att, v0_def):
    pos_obs_t, vel0, goal_pos_t = build_batch(df, windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(windows)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    with torch.no_grad():
        baseline_traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, zero, one, zero, one,
                                  tau, tau, v0_att, v0_def)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()

        fitted_traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, A1, B1, A2, B2,
                                tau, tau, v0_att, v0_def)
        fitted_loss = ((fitted_traj[..., :2] - pos_obs_t) ** 2).mean().item()

    improvement = (1 - fitted_loss / baseline_loss) * 100
    return dict(baseline_mse=baseline_loss, fitted_mse=fitted_loss, improvement_pct=improvement)


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    all_windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"total windows: {len(all_windows)}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(all_windows))
    test_idx, train_idx = perm[:N_TEST], perm[N_TEST:]
    train_windows = [all_windows[i] for i in train_idx]
    test_windows = [all_windows[i] for i in test_idx]
    print(f"train: {len(train_windows)}  test: {len(test_windows)}\n")

    pos_obs_t, vel0, goal_pos_t = build_batch(df, train_windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(train_windows)
    tau = torch.tensor(TAU_FIXED)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    # --- 段階1: train局面だけでv0を最適化 ---
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
    print(f"[train] drive-only v0: v0_att={v0_att_fixed.item():.3f} v0_def={v0_def_fixed.item():.3f}")

    # --- 段階2: v0固定、train局面だけでA,Bを最適化 ---
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
        if step % 500 == 0 or step == N_STEPS_STAGE2 - 1:
            print(f"[train stage2] step {step:4d}  loss={loss.item():.4f}  "
                  f"A1={A1.item():+.3f} B1={B1.item():.3f} A2={A2.item():+.3f} B2={B2.item():.3f}")

    A1_final, B1_final = raw_ab["A1"].detach(), torch.exp(raw_ab["log_B1"]).detach()
    A2_final, B2_final = raw_ab["A2"].detach(), torch.exp(raw_ab["log_B2"]).detach()
    print(f"\nfinal params: A1={A1_final.item():+.3f} B1={B1_final.item():.3f} "
          f"A2={A2_final.item():+.3f} B2={B2_final.item():.3f}")

    # --- 評価: train(in-sample)とtest(held-out)の両方 ---
    train_eval = eval_on(train_windows, attack_goal, df, tau, A1_final, B1_final, A2_final, B2_final,
                          v0_att_fixed, v0_def_fixed)
    test_eval = eval_on(test_windows, attack_goal, df, tau, A1_final, B1_final, A2_final, B2_final,
                         v0_att_fixed, v0_def_fixed)

    print(f"\n[train, in-sample]  baseline_RMSE={train_eval['baseline_mse']**0.5:.3f}m  "
          f"fitted_RMSE={train_eval['fitted_mse']**0.5:.3f}m  improvement={train_eval['improvement_pct']:.1f}%")
    print(f"[test,  held-out ]  baseline_RMSE={test_eval['baseline_mse']**0.5:.3f}m  "
          f"fitted_RMSE={test_eval['fitted_mse']**0.5:.3f}m  improvement={test_eval['improvement_pct']:.1f}%")

    out = {
        "tau_fixed": TAU_FIXED,
        "n_windows": N_WINDOWS,
        "n_test": N_TEST,
        "split_seed": SPLIT_SEED,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "v0_att_fixed": v0_att_fixed.item(),
        "v0_def_fixed": v0_def_fixed.item(),
        "final_params": {"A1": A1_final.item(), "B1": B1_final.item(),
                          "A2": A2_final.item(), "B2": B2_final.item()},
        "train_eval": train_eval,
        "test_eval": test_eval,
    }
    with open("scripts/held_out_validation_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/held_out_validation_result.json")


if __name__ == "__main__":
    main()
