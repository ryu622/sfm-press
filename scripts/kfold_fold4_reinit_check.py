"""kfold_cv.pyのfold=4だけを、A,Bの初期値を変えて複数回再実行する(追試17)。

追試16の5分割交差検証で、fold=4だけ突出して悪化した(test改善率-38.5%、
A1=11.08, B1=12.99, B2=0.013という他foldから外れた退化気味のパラメータ)。
この追試では、fold=4と全く同じtrain(32局面)/test(8局面)分割のまま、
段階2(A,Bの最適化)の初期値だけを5パターン変えて再実行し、毎回同じ
(悪い)局所解に収束するのか、初期値次第でtrain lossがより良い・
他foldに近い解に到達できるのかを確認する。段階1(v0の最適化)は
決定的な問題ではないと考え、fold=4の元の結果と同じ手続きのまま固定する。
"""

import json

import numpy as np
import torch

from scripts.real_data_pooled_fit import find_pooled_windows, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, build_snapshot, DT
from scripts.real_data_pooled_fit_v4 import build_batch
from scripts.tau_fixed_v4_fair_comparison import simulate
from scripts.held_out_validation import eval_on
from scripts.kfold_cv import TAU_FIXED, N_WINDOWS, K, N_STEPS_STAGE1, N_STEPS_STAGE2, SPLIT_SEED
from sfm_press.data import determine_attacking_goal_x

torch.set_default_dtype(torch.float64)

FOLD_ID = 4
INIT_SETS = [
    dict(A1=1.0, B1=3.0, A2=-1.0, B2=3.0, label="original(追試16と同じ)"),
    dict(A1=0.5, B1=5.0, A2=-0.5, B2=5.0, label="small/wide"),
    dict(A1=2.0, B1=2.0, A2=-2.0, B2=2.0, label="large/narrow"),
    dict(A1=4.0, B1=5.4, A2=-5.0, B2=3.2, label="他foldの典型値付近"),
    dict(A1=0.1, B1=8.0, A2=-0.1, B2=8.0, label="ほぼゼロ・広いB"),
]


def run_stage2(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, tau, v0_att_fixed, v0_def_fixed, init):
    raw_ab = {
        "A1": torch.tensor(init["A1"], requires_grad=True),
        "log_B1": torch.tensor(np.log(init["B1"]), requires_grad=True),
        "A2": torch.tensor(init["A2"], requires_grad=True),
        "log_B2": torch.tensor(np.log(init["B2"]), requires_grad=True),
    }
    opt2 = torch.optim.Adam(raw_ab.values(), lr=0.05)
    final_loss = None
    for step in range(N_STEPS_STAGE2):
        opt2.zero_grad()
        A1, B1 = raw_ab["A1"], torch.exp(raw_ab["log_B1"])
        A2, B2 = raw_ab["A2"], torch.exp(raw_ab["log_B2"])
        traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, A1, B1, A2, B2,
                         tau, tau, v0_att_fixed, v0_def_fixed)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        opt2.step()
        final_loss = loss.item()
    return (raw_ab["A1"].detach(), torch.exp(raw_ab["log_B1"]).detach(),
            raw_ab["A2"].detach(), torch.exp(raw_ab["log_B2"]).detach(), final_loss)


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    all_windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(all_windows))
    fold_size = len(all_windows) // K
    folds = [perm[i * fold_size:(i + 1) * fold_size] for i in range(K)]
    test_idx = folds[FOLD_ID]
    train_idx = np.concatenate([folds[j] for j in range(K) if j != FOLD_ID])
    train_windows = [all_windows[i] for i in train_idx]
    test_windows = [all_windows[i] for i in test_idx]
    print(f"fold={FOLD_ID}  n_train={len(train_windows)}  n_test={len(test_windows)}\n")

    pos_obs_t, vel0, goal_pos_t = build_batch(df, train_windows, attack_goal)
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(train_windows)
    tau = torch.tensor(TAU_FIXED)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    # 段階1(v0の最適化)はfold=4と同じ手続きで固定
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
    print(f"drive-only v0: v0_att={v0_att_fixed.item():.3f} v0_def={v0_def_fixed.item():.3f}\n")

    results = []
    for init in INIT_SETS:
        A1, B1, A2, B2, train_loss = run_stage2(
            pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, tau, v0_att_fixed, v0_def_fixed, init)
        test_eval = eval_on(test_windows, attack_goal, df, tau, A1, B1, A2, B2,
                             v0_att_fixed, v0_def_fixed)
        print(f"[init={init['label']:22s}] train_RMSE={train_loss**0.5:.3f}m  "
              f"A1={A1.item():+.3f} B1={B1.item():.3f} A2={A2.item():+.3f} B2={B2.item():.3f}  "
              f"test_improve={test_eval['improvement_pct']:.1f}%")
        results.append({
            "init_label": init["label"],
            "train_mse": train_loss,
            "final_params": {"A1": A1.item(), "B1": B1.item(), "A2": A2.item(), "B2": B2.item()},
            "test_improve": test_eval["improvement_pct"],
        })

    with open("scripts/kfold_fold4_reinit_result.json", "w") as f:
        json.dump({"fold_id": FOLD_ID, "results": results}, f)
    print("\nsaved: scripts/kfold_fold4_reinit_result.json")


if __name__ == "__main__":
    main()
