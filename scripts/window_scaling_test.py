"""局面数を増やすとA1,B1の尾根が狭まるかを検証する(real_data_fit_test.md 追試13)。

追試12で、τ=1.2秒固定・12局面プールの条件下では、3000ステップの最適化でも
B1が20〜300超の範囲を非収束的に振動し続けることを確認した(std/mean=0.501)。
これがデータ量不足(局面の距離配置・守備者数のバリエーションが乏しく、
複数守備者の力の総和が分離できない)によるものかを切り分けるため、
局面数を12→40に増やし(J03WPYから抽出できる候補48局面のうち上位40件)、
全く同じ手続き(段階1: v0のみ1000ステップ, 段階2: A,Bのみ3000ステップ,
τ=1.2秒固定)を再実行して、B1の尾根の広さを直接比較する。
"""

import json

import numpy as np
import torch

from scripts.real_data_pooled_fit import find_pooled_windows, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, build_snapshot
from scripts.real_data_pooled_fit_v4 import build_batch
from scripts.tau_fixed_v4_fair_comparison import simulate
from sfm_press.data import determine_attacking_goal_x

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_FIXED = 1.2
N_WINDOWS = 40
N_STEPS_STAGE1 = 1000
N_STEPS_STAGE2 = 3000


def run(df, windows, attack_goal, tau_fixed, n_steps_stage1, n_steps_stage2, verbose=True):
    pos_obs_t, vel0, goal_pos_t = build_batch(df, windows, attack_goal)
    from scripts.real_data_fit_test import DT
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(windows)

    tau_att = torch.tensor(tau_fixed)
    tau_def = torch.tensor(tau_fixed)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    # 段階1: v0だけ最適化
    raw_v0 = {
        "log_v0_att": torch.tensor(np.log(4.0), requires_grad=True),
        "log_v0_def": torch.tensor(np.log(2.0), requires_grad=True),
    }
    opt1 = torch.optim.Adam(raw_v0.values(), lr=0.05)
    for step in range(n_steps_stage1):
        opt1.zero_grad()
        v0_att = torch.exp(raw_v0["log_v0_att"])
        v0_def = torch.exp(raw_v0["log_v0_def"])
        traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, zero, one, zero, one,
                         tau_att, tau_def, v0_att, v0_def)
        loss = ((traj[..., :2] - pos_obs_t) ** 2).mean()
        loss.backward()
        opt1.step()
    v0_att_fixed = torch.exp(raw_v0["log_v0_att"]).detach()
    v0_def_fixed = torch.exp(raw_v0["log_v0_def"]).detach()
    with torch.no_grad():
        baseline_traj = simulate(pos_obs_t, vel0, goal_pos_t, t_eval, n_ep, zero, one, zero, one,
                                  tau_att, tau_def, v0_att_fixed, v0_def_fixed)
        baseline_loss = ((baseline_traj[..., :2] - pos_obs_t) ** 2).mean().item()
    print(f"drive-only baseline: RMSE={np.sqrt(baseline_loss):.3f}m  "
          f"(v0_att={v0_att_fixed.item():.3f}, v0_def={v0_def_fixed.item():.3f})\n")

    # 段階2: v0固定、A,Bだけ最適化
    raw_ab = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
    }
    opt2 = torch.optim.Adam(raw_ab.values(), lr=0.05)
    history = {"step": [], "loss": [], "A1": [], "B1": [], "A2": [], "B2": []}
    for step in range(n_steps_stage2):
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

        if verbose and (step % 200 == 0 or step == n_steps_stage2 - 1):
            print(f"step {step:4d}  loss={loss.item():.4f}  RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={A1.item():+.3f} B1={B1.item():.3f} A2={A2.item():+.3f} B2={B2.item():.3f}")

    tail = 500
    b1_tail = np.array(history["B1"][-tail:])
    a1_tail = np.array(history["A1"][-tail:])
    print(f"\nlast {tail} steps: B1 mean={b1_tail.mean():.2f} std={b1_tail.std():.2f} "
          f"(std/mean={b1_tail.std()/abs(b1_tail.mean()):.3f})")
    print(f"last {tail} steps: A1 mean={a1_tail.mean():.3f} std={a1_tail.std():.3f} "
          f"(std/mean={a1_tail.std()/abs(a1_tail.mean()):.3f})")

    final_loss = history["loss"][-1]
    improvement = (1 - final_loss / baseline_loss) * 100
    print(f"\nfinal RMSE={np.sqrt(final_loss):.3f}m  improvement={improvement:.1f}%")

    return {
        "n_windows": n_ep,
        "tau_fixed": tau_fixed,
        "n_steps_stage1": n_steps_stage1,
        "n_steps_stage2": n_steps_stage2,
        "v0_att_fixed": v0_att_fixed.item(),
        "v0_def_fixed": v0_def_fixed.item(),
        "baseline_mse": baseline_loss,
        "final_mse": final_loss,
        "improvement_pct": improvement,
        "history": history,
        "tail_stats": {
            "B1_mean": float(b1_tail.mean()), "B1_std": float(b1_tail.std()),
            "A1_mean": float(a1_tail.mean()), "A1_std": float(a1_tail.std()),
        },
    }


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows (requested {N_WINDOWS})\n")

    out = run(df, windows, attack_goal, TAU_FIXED, N_STEPS_STAGE1, N_STEPS_STAGE2)
    with open("scripts/window_scaling_test_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/window_scaling_test_result.json")


if __name__ == "__main__":
    main()
