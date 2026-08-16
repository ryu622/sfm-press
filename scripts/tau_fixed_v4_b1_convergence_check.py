"""B1(A1)の不安定性が「ステップ数不足」か「本質的な非収束」かを切り分ける(real_data_fit_test.md 追試12)。

追試11でτ=1.2秒の段階2において、B1が1000ステップ経っても148〜305の間で
激しく振動し続けていた。局面数(現在12)を増やす前に、まず安価な切り分けとして
τ=1.2秒に絞ってステップ数を3000まで伸ばし、それでも振動が収まらないかを確認する。

- 3000ステップでも振動が収まらない/減衰しないなら、局面数不足(単純にデータの
  制約が弱い)よりも、この駆動力設計・この局面群における本質的な非識別性の
  可能性が高まる。
- 振動が減衰し、どこかに収束するなら、単にステップ数が足りなかっただけ
  ということになる。
"""

import json

import numpy as np
import torch

from scripts.real_data_pooled_fit import find_pooled_windows, N_WINDOWS, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, build_snapshot
from scripts.real_data_pooled_fit_v4 import build_batch
from scripts.tau_fixed_v4_fair_comparison import simulate
from sfm_press.data import determine_attacking_goal_x

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

TAU_FIXED = 1.2
N_OPT_STEPS = 3000


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    pos_obs_t, vel0, goal_pos_t = build_batch(df, windows, attack_goal)
    from scripts.real_data_fit_test import DT
    t_eval = torch.linspace(0, (FIXED_LEN - 1) * DT, FIXED_LEN)
    n_ep = len(windows)

    tau_att = torch.tensor(TAU_FIXED)
    tau_def = torch.tensor(TAU_FIXED)
    zero, one = torch.tensor(0.0), torch.tensor(1.0)

    # 段階1: v0だけ最適化(追試11と同じ手順)
    raw_v0 = {
        "log_v0_att": torch.tensor(np.log(4.0), requires_grad=True),
        "log_v0_def": torch.tensor(np.log(2.0), requires_grad=True),
    }
    opt1 = torch.optim.Adam(raw_v0.values(), lr=0.05)
    for step in range(1000):
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

    # 段階2: v0固定、A,Bだけ最適化(3000ステップ)
    raw_ab = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(3.0), requires_grad=True),
        "A2": torch.tensor(-1.0, requires_grad=True),
        "log_B2": torch.tensor(np.log(3.0), requires_grad=True),
    }
    opt2 = torch.optim.Adam(raw_ab.values(), lr=0.05)
    history = {"step": [], "loss": [], "A1": [], "B1": [], "A2": [], "B2": []}
    for step in range(N_OPT_STEPS):
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

        if step % 200 == 0 or step == N_OPT_STEPS - 1:
            print(f"step {step:4d}  loss={loss.item():.4f}  RMSE={np.sqrt(loss.item()):.3f}m  "
                  f"A1={A1.item():+.3f} B1={B1.item():.3f} A2={A2.item():+.3f} B2={B2.item():.3f}")

    # 収束判定: 最後の500ステップでのB1, A1の変動係数(std/mean)を見る
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

    out = {
        "tau_fixed": TAU_FIXED,
        "n_opt_steps": N_OPT_STEPS,
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
    with open("scripts/tau_fixed_v4_b1_convergence_result.json", "w") as f:
        json.dump(out, f)
    print("\nsaved: scripts/tau_fixed_v4_b1_convergence_result.json")


if __name__ == "__main__":
    main()
