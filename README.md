# sfm-press

社会力モデル(Social Force Model, SFM; Helbing & Molnár, 1995)を、サッカーの守備プレッシャーの定量化に道具的に援用する修士研究のリポジトリ。

SFMを「プレッシャーそのもの」と主張するのではなく、駆動力(意図)+相互作用力(近くの相手から受ける力)という解釈可能なパラメータ体系を持つ既存の枠組みを使って、選手軌道をどれだけ説明できるか、そこから導出される係数がプレッシャー概念とどれだけ整合するかを検証する立場を取る。詳細は `documents/research_plan.md` を参照。

## 現状(2026-08-17時点)

実データでの軌道フィッティングを重ねた結果、相互作用項には一貫して正の点推定(改善率数%〜二桁%)が見られる一方、最も厳格な検証(5分割交差検証)では統計的に有意ではないことが判明した。加えて、駆動力項だけのbaselineモデル自体の当てはまりも良くなく、SFMの前提とする歩行者群集の行動様式とサッカーの守備行動との間に構造的なミスマッチがある可能性が浮上している。

現時点の評価とテーマ継続の是非については `documents/theme_viability_assessment.md` を、検証の全過程(追試1〜18)は `documents/real_data_fit_test.md` を参照。

## 環境構築

`uv` でPython 3.12の仮想環境を管理する(CLAUDE.md参照)。

```bash
uv sync
```

トラッキングデータは [idsse-data](https://huggingface.co/datasets/idsse/idsse-data)(Sportec/ブンデスリーガ、Hugging Face経由)を `kloppy` で読み込む。初回実行時に `data/cache/` へパース済みデータをキャッシュする。

## ディレクトリ構成

```
sfm_press/          コアパッケージ(データパイプライン・可視化)
  data.py             kloppyでのトラッキングデータ読み込み、SG平滑化、キャッシュ
  viz.py              mplsoccerによる可視化

scripts/             各検証・実験スクリプト(追試ごとに独立、*_result.json に結果を保存)
documents/           研究計画書・各検証のまとめ(実験ログ)・図表
  research_plan.md              研究計画書(目的・先行研究・検証計画)
  theme_viability_assessment.md テーマ継続可否の現状評価(随時更新)
  real_data_fit_test.md         実データでの軌道フィッティング検証ログ(追試1〜)
  phase1_data_pipeline.md       データパイプラインの基本統計量
  check_distance_range.md       距離レンジ・識別性の初期判定
  synthetic_recovery_test.md    合成データでの係数復元テスト
  noise_robustness_test.md      ノイズ頑健性テスト
  figures/                      各検証の図(PNG)
```

## 実行方法

スクリプトは `sfm_press` パッケージや他の `scripts/*` を相互にimportするため、モジュールとして実行する(直接 `python scripts/foo.py` すると `ModuleNotFoundError` になる)。

```bash
uv run python -m scripts.<script_name>
```

例:

```bash
# データパイプラインの基本統計量を可視化
uv run python -m scripts.basic_stats

# 実データでの軌道フィッティング(局面数40、フェアな2段階最適化)
uv run python -m scripts.window_scaling_test
```

## 開発ガイドライン

`pip` / `python` を直接使わず、常に `uv run` / `uv add` を介して操作する。詳細は `CLAUDE.md` を参照。
