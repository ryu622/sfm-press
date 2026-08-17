"""held-out改善率が分割の運によるものでないかを、複数の乱数分割で確認する(追試15)。

追試14はseed=0の1回のtrain30/test10分割のみで、test改善率(13.4%)がtrain(8.5%)を
上回るという結果だったが、これが「たまたま良い分割を引いた」だけなのか、
複数の分割で安定して見られる傾向なのかは未確認だった。この追試ではseedを
0〜4の5通り試し、毎回のtrain/test改善率を記録する。
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
N_TEST = 10
N_STEPS_STAGE1 = 1000
N_STEPS_STAGE2 = 3000
SEEDS = [0, 1, 2, 3, 4]


def run_one_split(df, all_windows, attack_goal, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_windows))
    test_idx, train_idx = perm[:N_TEST], perm[N_TEST:]
    train_windows = [all_windows[i] for i in train_idx]
    test_windows = [all_windows[i] for i in test_idx]

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

    train_eval = eval_on(train_windows, attack_goal, df, tau, A1_final, B1_final, A2_final, B2_final,
                          v0_att_fixed, v0_def_fixed)
    test_eval = eval_on(test_windows, attack_goal, df, tau, A1_final, B1_final, A2_final, B2_final,
                         v0_att_fixed, v0_def_fixed)

    print(f"[seed={seed}] train_improve={train_eval['improvement_pct']:.1f}%  "
          f"test_improve={test_eval['improvement_pct']:.1f}%  "
          f"A1={A1_final.item():+.3f} B1={B1_final.item():.3f} "
          f"A2={A2_final.item():+.3f} B2={B2_final.item():.3f}")

    return {
        "seed": seed,
        "train_improve": train_eval["improvement_pct"],
        "test_improve": test_eval["improvement_pct"],
        "final_params": {"A1": A1_final.item(), "B1": B1_final.item(),
                          "A2": A2_final.item(), "B2": B2_final.item()},
    }


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    all_windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"total windows: {len(all_windows)}\n")

    results = [run_one_split(df, all_windows, attack_goal, seed) for seed in SEEDS]

    train_vals = [r["train_improve"] for r in results]
    test_vals = [r["test_improve"] for r in results]
    print(f"\ntrain improve: mean={np.mean(train_vals):.1f}%  std={np.std(train_vals):.1f}%")
    print(f"test  improve: mean={np.mean(test_vals):.1f}%  std={np.std(test_vals):.1f}%")
    print(f"test improve range: [{min(test_vals):.1f}%, {max(test_vals):.1f}%]")

    with open("scripts/held_out_validation_multiseed_result.json", "w") as f:
        json.dump({"seeds": SEEDS, "results": results}, f)
    print("\nsaved: scripts/held_out_validation_multiseed_result.json")


if __name__ == "__main__":
    main()
