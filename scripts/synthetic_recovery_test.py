"""合成データによるSFMパラメータ回復性テスト(research_plan.md 5.1節の懸念に対する事前チェック)。

結果は scripts/synthetic_recovery_result.json に保存し、documents/ 側の
まとめドキュメントで参照する(plot_synthetic_recovery.py で図を生成)。

最小構成: 攻撃者(ボール保持者)1人 + 守備者N人。
- 攻撃者は前進方向へ駆動力を持ち、守備者集団からの力(A1, B1)を受ける。
- 守備者は「自陣方向への後退」という弱い駆動力を持ちつつ、攻撃者からの力(A2, B2, 符号は負=引力を想定)を受ける。
- 複数の守備者からの力は攻撃者側で総和される(3.3節の sum_j f_ij に対応)ため、
  この「重ね合わせ」がある状態でも(B)の軌道ベース最適化がA1,B1,A2,B2,tau_att,tau_defを
  復元できるかを検証する。

実データではなく完全に合成データ(ノイズなし)で行う理想条件でのテストである点に注意。
"""

import json
import time

import numpy as np
import torch
from torchdiffeq import odeint

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)

device = torch.device("cpu")

N_DEF = 3
N_EPISODES = 20
T = 6.0
FPS = 25
N_STEPS = int(T * FPS) + 1
T_EVAL = torch.linspace(0, T, N_STEPS)

# --- 真のパラメータ ---
TRUE = dict(
    A1=3.0,   # 攻撃者が守備者集団から受ける力の強さ(正=斥力/回避)
    B1=4.0,   # 同上、到達距離
    A2=-4.0,  # 守備者が攻撃者から受ける力の強さ(負=引力/チェイス)
    B2=6.0,   # 同上、到達距離
    tau_att=0.5,
    tau_def=1.0,
)
V0_ATT = 5.0
V0_DEF = 1.5
E_ATT = torch.tensor([1.0, 0.0])   # 攻撃者は常に+x方向へ進もうとする
E_DEF = torch.tensor([-1.0, 0.0])  # 守備者は放っておくと自陣(-x方向)へ戻ろうとする
R0 = 0.3  # r_ij(選手の実効半径和相当)、固定・既知とする


def make_initial_states(n_episodes: int, n_def: int) -> torch.Tensor:
    """state shape: (E, 1+N_def, 4) = [x, y, vx, vy]"""
    state = torch.zeros(n_episodes, 1 + n_def, 4)
    # 攻撃者: 原点付近からスタート、初速はドライブ方向
    state[:, 0, 0] = torch.rand(n_episodes) * 4 - 2
    state[:, 0, 1] = torch.rand(n_episodes) * 4 - 2
    state[:, 0, 2:4] = E_ATT * V0_ATT * 0.5

    # 守備者: 攻撃者から10〜25mの範囲にランダムな角度で配置
    for k in range(n_def):
        ang = torch.rand(n_episodes) * 2 * np.pi
        dist = torch.rand(n_episodes) * 15 + 10
        state[:, 1 + k, 0] = state[:, 0, 0] + dist * torch.cos(ang)
        state[:, 1 + k, 1] = state[:, 0, 1] + dist * torch.sin(ang)
        state[:, 1 + k, 2:4] = 0.0
    return state


def social_force_ode(A1, B1, A2, B2, tau_att, tau_def):
    def f(t, state):
        pos = state[..., :2]
        vel = state[..., 2:4]

        att_pos = pos[:, 0, :]          # (E,2)
        def_pos = pos[:, 1:, :]         # (E,Ndef,2)
        att_vel = vel[:, 0, :]
        def_vel = vel[:, 1:, :]

        drive_att = (V0_ATT * E_ATT - att_vel) / tau_att            # (E,2)
        drive_def = (V0_DEF * E_DEF - def_vel) / tau_def            # (E,Ndef,2)

        # 攻撃者 <- 守備者集団(総和)
        diff1 = att_pos.unsqueeze(1) - def_pos                      # (E,Ndef,2) defender->attacker
        dist1 = diff1.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        n1 = diff1 / dist1
        f_mag1 = A1 * torch.exp((R0 - dist1) / B1)
        f_att_from_def = (f_mag1 * n1).sum(dim=1)                   # (E,2)

        # 各守備者 <- 攻撃者
        diff2 = def_pos - att_pos.unsqueeze(1)                      # (E,Ndef,2) attacker->defender
        dist2 = diff2.norm(dim=-1, keepdim=True).clamp(min=1e-2)
        n2 = diff2 / dist2
        f_mag2 = A2 * torch.exp((R0 - dist2) / B2)
        f_def_from_att = f_mag2 * n2                                 # (E,Ndef,2)

        accel_att = drive_att + f_att_from_def
        accel_def = drive_def + f_def_from_att

        dpos_att = att_vel
        dpos_def = def_vel

        d_att = torch.cat([dpos_att, accel_att], dim=-1).unsqueeze(1)   # (E,1,4)
        d_def = torch.cat([dpos_def, accel_def], dim=-1)                 # (E,Ndef,4)
        return torch.cat([d_att, d_def], dim=1)

    return f


def simulate(params: dict, y0: torch.Tensor) -> torch.Tensor:
    f = social_force_ode(**params)
    traj = odeint(f, y0, T_EVAL, method="rk4", options={"step_size": 0.02})
    return traj  # (N_STEPS, E, 1+Ndef, 4)


def main():
    y0 = make_initial_states(N_EPISODES, N_DEF)

    print("=== ground truth simulation ===")
    with torch.no_grad():
        traj_true = simulate(TRUE, y0)
    pos_obs = traj_true[..., :2].detach()

    dists = (pos_obs[:, :, 0:1, :] - pos_obs[:, :, 1:, :]).norm(dim=-1)
    print(f"synthetic distance range: min={dists.min():.2f}m max={dists.max():.2f}m "
          f"median={dists.median():.2f}m")

    # --- 推定対象パラメータ(意図的に真値からずらして初期化) ---
    raw = {
        "A1": torch.tensor(1.0, requires_grad=True),
        "log_B1": torch.tensor(np.log(2.0), requires_grad=True),
        "A2": torch.tensor(1.0, requires_grad=True),  # 符号もわざと逆から始める
        "log_B2": torch.tensor(np.log(2.0), requires_grad=True),
        "log_tau_att": torch.tensor(np.log(1.0), requires_grad=True),
        "log_tau_def": torch.tensor(np.log(2.0), requires_grad=True),
    }

    def get_params():
        return dict(
            A1=raw["A1"],
            B1=torch.exp(raw["log_B1"]),
            A2=raw["A2"],
            B2=torch.exp(raw["log_B2"]),
            tau_att=torch.exp(raw["log_tau_att"]),
            tau_def=torch.exp(raw["log_tau_def"]),
        )

    optimizer = torch.optim.Adam(raw.values(), lr=0.08)

    print("\n=== optimization ===")
    n_steps = 300
    history = {"step": [], "loss": [], **{k: [] for k in TRUE}}
    t0 = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        params = get_params()
        traj_pred = simulate(params, y0)
        loss = ((traj_pred[..., :2] - pos_obs) ** 2).mean()
        loss.backward()
        optimizer.step()

        p = {k: v.item() for k, v in params.items()}
        history["step"].append(step)
        history["loss"].append(loss.item())
        for k in TRUE:
            history[k].append(p[k])

        if step % 20 == 0 or step == n_steps - 1:
            print(
                f"step {step:4d}  loss={loss.item():.5f}  "
                f"A1={p['A1']:+.3f} B1={p['B1']:.3f} "
                f"A2={p['A2']:+.3f} B2={p['B2']:.3f} "
                f"tau_att={p['tau_att']:.3f} tau_def={p['tau_def']:.3f}"
            )
    print(f"elapsed: {time.time() - t0:.1f}s")

    print("\n=== final comparison (true vs recovered) ===")
    final = {k: v.item() for k, v in get_params().items()}
    summary = {}
    for k in TRUE:
        true_v = TRUE[k]
        est_v = final[k]
        rel_err = abs(est_v - true_v) / (abs(true_v) + 1e-8) * 100
        sign_ok = np.sign(true_v) == np.sign(est_v) or true_v == 0
        summary[k] = {"true": true_v, "recovered": est_v, "rel_err_pct": rel_err, "sign_ok": bool(sign_ok)}
        print(f"  {k:10s}  true={true_v:+.3f}  recovered={est_v:+.3f}  rel_err={rel_err:5.1f}%  "
              f"{'OK' if sign_ok else 'SIGN FLIPPED'}")

    out = {
        "n_episodes": N_EPISODES,
        "n_def": N_DEF,
        "true_params": TRUE,
        "init_params": {k: (v.item() if "log" not in k else float(np.exp(v.item())))
                         for k, v in raw.items()},
        "distance_range": {
            "min": float(dists.min()), "max": float(dists.max()), "median": float(dists.median())
        },
        "history": history,
        "summary": summary,
        "elapsed_sec": time.time() - t0,
    }
    with open("scripts/synthetic_recovery_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved: scripts/synthetic_recovery_result.json")


if __name__ == "__main__":
    main()
