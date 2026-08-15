"""τの下限制約を緩めた感度分析(real_data_fit_test.md 追試6)。

追試5でEMA半減期を伸ばしてもτは常に下限(0.3秒)に張り付くことが分かった。
これが「τの真の値は小さいが有限のどこかで自然に収まる」のか、
「τが際限なく0へ向かい続ける(モデルの表現力不足の代償)」のかを切り分けるため、
τの下限をさらに下げた場合にどこで収束するかを確認する。
半減期は追試4-5で最もフィットが良かった0.3秒に固定する。
"""

import json

from scripts.real_data_fit_test import MATCH_ID, build_snapshot
from scripts.real_data_pooled_fit import FIXED_LEN, N_WINDOWS, find_pooled_windows
from scripts.real_data_pooled_fit_v3 import EMA_HALFLIFE_SEC, TAU_MAX, run_experiment

TAU_MIN_LEVELS = [0.3, 0.15, 0.05, 0.01]
N_OPT_STEPS = 1200


def main():
    df = build_snapshot(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    results = []
    for tau_min in TAU_MIN_LEVELS:
        out = run_experiment(df, windows, EMA_HALFLIFE_SEC, n_opt_steps=N_OPT_STEPS, verbose=True,
                              tau_min=tau_min, tau_max=TAU_MAX)
        results.append(out)
        print()

    print("=== summary across tau floors ===")
    header = f"{'tau_min[s]':>10s}  {'RMSE[m]':>8s}  {'improve%':>9s}  "
    header += "  ".join(f"{k:>10s}" for k in ["A1", "B1", "A2", "B2", "tau_att", "tau_def"])
    print(header)
    for r in results:
        p = r["final_params"]
        row = f"{r['tau_min']:>10.3f}  {r['final_mse']**0.5:>8.3f}  {r['improvement_pct']:>9.1f}  "
        row += "  ".join(f"{p[k]:>10.3f}" for k in ["A1", "B1", "A2", "B2", "tau_att", "tau_def"])
        print(row)
        pinned = []
        if abs(p["tau_att"] - r["tau_min"]) < 0.005:
            pinned.append("tau_att@MIN")
        if abs(p["tau_def"] - r["tau_min"]) < 0.005:
            pinned.append("tau_def@MIN")
        if pinned:
            print(f"  -> pinned: {', '.join(pinned)}")

    with open("scripts/tau_floor_sweep_result.json", "w") as f:
        json.dump({"tau_min_levels": TAU_MIN_LEVELS, "results": results}, f)
    print("\nsaved: scripts/tau_floor_sweep_result.json")


if __name__ == "__main__":
    main()
