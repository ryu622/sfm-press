"""動的ポテンシャル駆動力(v4)でのτ固定診断テスト(real_data_fit_test.md 追試10続き)。

real_data_pooled_fit_v4.py でτを自由にすると上限(1.2秒)に張り付いた
(因果的EMA版でτが下限に張り付いたのと逆方向)。これがどこまで意味のある
収束かを確認するため、τを複数の固定値で振り、A,B(およびv0)がどれだけ
説明力を持つかを診断する。
"""

import json

from scripts.real_data_pooled_fit import find_pooled_windows, N_WINDOWS, FIXED_LEN
from scripts.real_data_fit_test import MATCH_ID, build_snapshot
from scripts.real_data_pooled_fit_v4 import run_experiment
from sfm_press.data import determine_attacking_goal_x

TAU_FIXED_LEVELS = [0.5, 1.2, 2.0, 3.0]
N_OPT_STEPS = 1000


def main():
    df = build_snapshot(MATCH_ID)
    attack_goal = determine_attacking_goal_x(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    results = []
    for tau_fixed in TAU_FIXED_LEVELS:
        print(f"\n########## tau_fixed = {tau_fixed}s ##########")
        out = run_experiment(df, windows, attack_goal, n_opt_steps=N_OPT_STEPS, verbose=True,
                              tau_fixed=tau_fixed)
        results.append(out)

    print("\n=== summary across fixed tau (v4 drive) ===")
    header = f"{'tau_fixed[s]':>12s}  {'RMSE[m]':>8s}  {'improve%':>9s}  "
    header += "  ".join(f"{k:>9s}" for k in ["A1", "B1", "A2", "B2", "v0_att", "v0_def"])
    print(header)
    for r in results:
        p = r["final_params"]
        row = f"{r['tau_fixed']:>12.2f}  {r['final_mse']**0.5:>8.3f}  {r['improvement_pct']:>9.1f}  "
        row += "  ".join(f"{p[k]:>9.3f}" for k in ["A1", "B1", "A2", "B2", "v0_att", "v0_def"])
        print(row)

    with open("scripts/tau_fixed_v4_sweep_result.json", "w") as f:
        json.dump({"tau_fixed_levels": TAU_FIXED_LEVELS, "results": results}, f)
    print("\nsaved: scripts/tau_fixed_v4_sweep_result.json")


if __name__ == "__main__":
    main()
