"""合成データパラメータ回復テストに観測ノイズを加えたロバスト性チェック。

synthetic_recovery_test.py の run_experiment() を複数のノイズレベルで実行し、
実データのトラッキングノイズを想定した条件でも(B)の軌道ベース最適化が
妥当な精度でパラメータを復元できるかを確認する。
"""

import json

from synthetic_recovery_test import run_experiment

# 光学トラッキング(TRACAB等)の一般的な精度を踏まえ、楽観〜悲観的な水準を含めて設定 [m]
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
N_STEPS = 300


def main():
    results = []
    for noise_std in NOISE_LEVELS:
        print(f"\n########## noise_std = {noise_std} m ##########")
        out = run_experiment(noise_std=noise_std, seed=0, n_steps=N_STEPS, verbose=True)
        results.append(out)

    print("\n=== summary across noise levels ===")
    print(f"{'noise[m]':>9s}  " + "  ".join(f"{k:>16s}" for k in results[0]["summary"]))
    for r in results:
        row = f"{r['noise_std']:>9.2f}  "
        row += "  ".join(f"{r['summary'][k]['recovered']:+16.3f}" for k in r["summary"])
        print(row)

    with open("scripts/noise_robustness_result.json", "w") as f:
        json.dump({"noise_levels": NOISE_LEVELS, "results": results}, f, indent=2)
    print("\nsaved: scripts/noise_robustness_result.json")


if __name__ == "__main__":
    main()
