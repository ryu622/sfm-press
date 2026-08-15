"""因果的EMA半減期の感度分析(real_data_fit_test.md 追試5)。

real_data_pooled_fit_v3.py で tau_att, tau_def が制約下限(0.3秒)に張り付いた。
これがEMA半減期(0.3秒)自体が短すぎることに起因するかを確認するため、
複数の半減期で同じ12局面プール推定を実行し、tauが制約の内側で
自然に収束するようになるか、A,Bの値がどれだけ半減期に敏感かを見る。
"""

import json

from scripts.real_data_fit_test import MATCH_ID, build_snapshot
from scripts.real_data_pooled_fit import FIXED_LEN, N_WINDOWS, find_pooled_windows
from scripts.real_data_pooled_fit_v3 import TAU_MAX, TAU_MIN, run_experiment

HALFLIFE_LEVELS = [0.3, 0.5, 0.8, 1.2, 2.0]
N_OPT_STEPS = 1200


def main():
    df = build_snapshot(MATCH_ID)
    windows = find_pooled_windows(df, N_WINDOWS, FIXED_LEN)
    print(f"selected {len(windows)} windows\n")

    results = []
    for halflife in HALFLIFE_LEVELS:
        out = run_experiment(df, windows, halflife, n_opt_steps=N_OPT_STEPS, verbose=True)
        results.append(out)
        print()

    print("=== summary across EMA halflives ===")
    header = f"{'halflife[s]':>11s}  {'RMSE[m]':>8s}  {'improve%':>9s}  "
    header += "  ".join(f"{k:>10s}" for k in ["A1", "B1", "A2", "B2", "tau_att", "tau_def"])
    print(header)
    for r in results:
        p = r["final_params"]
        row = f"{r['ema_halflife_sec']:>11.2f}  {r['final_mse']**0.5:>8.3f}  {r['improvement_pct']:>9.1f}  "
        row += "  ".join(f"{p[k]:>10.3f}" for k in ["A1", "B1", "A2", "B2", "tau_att", "tau_def"])
        print(row)
        pinned = []
        if abs(p["tau_att"] - TAU_MIN) < 0.01:
            pinned.append("tau_att@MIN")
        if abs(p["tau_def"] - TAU_MIN) < 0.01:
            pinned.append("tau_def@MIN")
        if abs(p["tau_att"] - TAU_MAX) < 0.01:
            pinned.append("tau_att@MAX")
        if abs(p["tau_def"] - TAU_MAX) < 0.01:
            pinned.append("tau_def@MAX")
        if pinned:
            print(f"  -> pinned: {', '.join(pinned)}")

    with open("scripts/ema_halflife_sweep_result.json", "w") as f:
        json.dump({"halflife_levels": HALFLIFE_LEVELS, "results": results}, f)
    print("\nsaved: scripts/ema_halflife_sweep_result.json")


if __name__ == "__main__":
    main()
