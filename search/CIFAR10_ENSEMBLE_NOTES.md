# CIFAR-10 化とアンサンブル探索：作業記録と論文用メモ

最終更新: 2026-09-29

ProxylessNAS を CIFAR-10 / RTX 5080 で動かすための変更、アンサンブル用に追加した機能、現時点の実験結果、論文に書くべき引用と手法上の注意点をまとめたもの。

---

## 1. 実行環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA GeForce RTX 5080（Blackwell, sm_120, 16GB）, driver 580 |
| venv | `env5080`（Python 3.12） |
| PyTorch | 2.11.0+cu128 / torchvision 0.26.0+cu128（`https://download.pytorch.org/whl/cu128` から導入） |
| その他 | PyYAML 導入、torchaudio 削除（未使用） |

- RTX 5080 には **cu128 以降のビルドが必須**。torch 2.6.0+cu124 はインストール自体は成功するが、GPU 演算が `no kernel image is available` で失敗する。
- 新しい PyTorch では `torch.optim.Adam` の `betas` 型チェックが厳しいため、`--arch_adam_beta1` の既定値を `0` → `0.0` に変更した。
- 既知の問題：1 回だけ `libnvrtc.so.12` 内でセグメンテーション違反が発生（試運転 7 回中 1 回、再現せず）。`cudnn.benchmark=True`（[run_manager.py](run_manager.py)）時の実行時カーネルコンパイルが疑わしい。再発したら `--resume` で再開し、頻発するなら benchmark 無効化か cu130 版を試す。

## 2. CIFAR-10 対応の変更点

コミット `a6835ae`、および `d3dea54` / `05c94c0`（作業者コミット）に含まれる。

| ファイル | 変更内容 |
|---|---|
| [models/super_nets/super_proxyless.py](models/super_nets/super_proxyless.py) | `first_stride` 引数を追加（既定 2 = ImageNet と同じ） |
| [cifar10_arch_search.py](cifar10_arch_search.py) | stem stride 1、`stride_stages` 既定を `1,2,2,1,2,1` に変更（特徴マップ 32→32→16→8→8→4→4。旧設定では最終 1×1）。FLOPs 基準値を再計算。`--target_hardware` を `flops` / `None` に限定 |
| [run_manager.py](run_manager.py) | mobile レイテンシ予測を 224×224 以外の入力で明示的にエラーにする |
| [cifar10_run_exp.py](cifar10_run_exp.py) | 末尾で欠落していた検証・テスト評価を追加（結果を `<path>/output` に保存） |
| [configs/config_5080.yaml](configs/config_5080.yaml) | compute capability 12.0、メモリ 16GB に修正 |

### FLOPs 基準値の根拠
公式の proxyless_mobile / gpu / cpu を、ImageNet 設定（224 入力、stem stride 2）と上記 CIFAR 設定（32 入力、stem stride 1、第 1 段 stride 1）で比べると、FLOPs 比はそれぞれ 0.302 / 0.298 / 0.285 だった。そこで ImageNet の基準値に 0.30 を掛けた（1.00× で 300M → 90M）。

### mobile レイテンシを使わない理由
公式の mobile レイテンシ表（`mobile_trim.yaml`）は 224×224 入力の ImageNet 形状でしか計測されていない。32×32 入力の値は存在しないため、CIFAR-10 ではハードウェア制約として FLOPs のみを使う。

## 3. アンサンブル用に追加したスクリプト

### 3.1 [cifar10_derive_archs.py](cifar10_derive_archs.py)：1 回の探索から複数構造を取り出す
探索のチェックポイントのアーキテクチャパラメータ α から K 個の構造を取り出し、`learned_net` と同じ形式（`net.config` / `run.config` / `init`）で書き出す。

- 候補の生成（`--mode`）
  - `topk`：同時確率 ∏ softmax(α) の上位 K 個（ビームサーチで厳密に計算）
  - `sample`：softmax(α / T) からサンプリング
- 絞り込み：`--min_flops` / `--max_flops`、`--min_hamming`（構造どうしの最小相違ブロック数）
- 選択（`--select`）
  - `order`：条件を満たした候補を先頭から K 個
  - `dpp`：`--pool_size` 個の候補を評価し、NEAS の Diversity Score で K 個を選ぶ（下記）
- `arch_00` は常に最も確率の高い構造（`--no_argmax` で解除）。既存の `learned_net/net.config` と一致することを確認済み。

**`--select dpp` の処理**（NEAS [4] の Eq.4, 5, 7, 8 をモデル単位に適用）

1. 各候補を supernet の継承重みで作り、BN 統計を学習データで再計算（NEAS [4]、SPOS [9] と同様）
2. 品質 r_p：学習データのうち、検証用 5,000 枚とは別の 2,000 枚での top-1 精度
3. 類似度 S_pq = E_n[ exp(−β ‖v_p(n) − v_q(n)‖₂) ]。v は最終層直前の 1280 次元特徴。β の既定は全ペアの平均距離の中央値の逆数
4. L = diag(r^γ) · S · diag(r^γ) として、det(L_部分集合) が最大になる K 個を貪欲法で選ぶ

### 3.2 [cifar10_ensemble_eval.py](cifar10_ensemble_eval.py)：アンサンブルの評価
- 各モデルの softmax 出力を平均する（Deep Ensembles [10] と同じ平均方式）
- 出力する指標：各モデルの精度と NLL、先頭 m 個のアンサンブルの精度（m=1…K）、oracle 精度、予測の食い違い率（ペアごと）、合計 FLOPs・合計パラメータ数
- `--split valid` では学習時と同じ固定 5,000 枚を使う。組み合わせの決定は valid、最終報告は test と使い分ける

### 実行例
```bash
source env5080/bin/activate
# 構造の取り出し
python search/cifar10_derive_archs.py --search_path Exp2/proxyless_gpu --out_path Exp2/ensemble_dpp \
  --n_archs 3 --mode sample --select dpp --pool_size 30
# 各構造の再学習
for d in Exp2/ensemble_dpp/arch_*; do
  python search/cifar10_run_exp.py --path $d --train --gpu 0 --hw-config search/configs/config_5080.yaml > $d/train.log 2>&1
done
# 評価
python search/cifar10_ensemble_eval.py --paths Exp2/ensemble_dpp/arch_* --output Exp2/ensemble_dpp/ensemble_test.json
```

注意：`cifar10_run_exp.py` の `--path` には `net.config` があるフォルダを指定すること。探索の出力フォルダ（例：`Exp2/proxyless_gpu`）を指定すると、ImageNet 用の構造と重みが読み込まれて size mismatch になる。

## 4. 現時点の実験結果

### 4.1 アーキテクチャ探索（`Exp2/proxyless_gpu`）
- 設定：ウォームアップ 20 エポック + 探索 150 エポック、batch 256、lr 0.05（cosine）、dropout 0.3、`--target_hardware None`、勾配法（`full_v2`）
- 所要時間：約 65 分（ウォームアップ約 8 分 + 探索約 57 分、ログの時刻と 1 反復あたりの時間から推定）

| 時点 | supernet 検証 top-1 |
|---|---|
| ウォームアップ 20/20 | 20.5% |
| 探索 150/150 | 58.3%（最高 58.6%） |

- 探索終了時のαのエントロピーは 1 ブロックあたり平均 1.62（全 21 ブロック合計 34.1）。一様分布なら平均 1.90（候補 7 個のブロックは ln 7、6 個のブロックは ln 6）なので、確率分布はまだほぼ一様
- 得られた構造：163.0M FLOPs、3.28M パラメータ

### 4.2 探索で得た構造の再学習（`Exp2/cifar10_retrain`）
- 設定：`learned_net/run.config` を使用（150 エポック、lr 0.05、label smoothing 0.1、wd 4e-5）。supernet の継承重みから学習開始
- 所要時間：約 45 分

| | top-1 |
|---|---|
| 学習前（継承重みのまま） | テスト 59.4% |
| 再学習後（150 エポック） | **検証 89.16%（最高 89.46%）/ テスト 89.10%** |

→ 単体モデルのベースライン：**テスト 89.10%、163.0M FLOPs、3.28M パラメータ**

### 4.3 構造の取り出し（動作確認のみ、再学習は未実施）
- `topk`：最も確率の高い構造から 1〜2 ブロックしか違わない。`--min_hamming 4` で 4 ブロック以上違う組み合わせにできる
- `sample`：最も確率の高い構造から 12〜16 ブロック違い、ほぼランダム
- `dpp`（候補 12 個から 3 個）：選ばれた 3 構造の類似度は 0.37〜0.39（候補全体では 0.32〜0.60）。類似度が最も低い候補（0.32）は品質 56% と低いため選ばれず、品質と多様性の両方が考慮されている
- 継承重みのまま 3 構造をアンサンブルしても、テスト精度 59.0%（単体 59.4%）で向上しない。重みを共有しているため誤りも似ていると考えられ、効果は再学習後に評価する必要がある

## 5. 未解決の課題

1. **supernet の学習が極端に遅い**：ウォームアップ 20 エポックで 20%、最終 58%。一方、固定ネットワークは同条件で 1 エポック目に 62%、最終 89%。原因はデータやハイパーパラメータより、supernet の学習手順（経路のランダム選択）側にある可能性が高い。α の収束と品質 r の推定精度に直結するので、最優先で調べる
2. `topk` と `dpp` の比較（再学習してアンサンブル精度で比べる）が未実施
3. `cifar10_run_exp.py --net proxyless_*`（`net.config` がないフォルダを指定した場合）は ImageNet 用の公式構造を読み込む。探索結果を使う通常の流れには影響しない
4. ログの `MixedEdge do not support set_arch_param_grad()` は、Zero が選ばれた回に `AP_path_wb.grad` が空のため出るもので、元の実装と同じ挙動（実害なし）

## 6. 論文に書くべき点

### 6.1 元の手法からの変更（手法・実験設定の節に明記）
- **CIFAR-10 向けの supernet 変更**：stem stride 2→1、第 1 段 stride 2→1（最終特徴マップ 4×4）
- **ハードウェア制約**：mobile レイテンシ表は 224 入力専用なので FLOPs を使用。基準値は公式構造の FLOPs 比（0.28〜0.30）から ImageNet の値を換算
- **NEAS の Diversity Score の転用**：NEAS [4] は層ごとの演算の組み合わせを評価して、supernet 学習中に探索空間を縮める。本研究では同じ DPP カーネルを**構造単位**で使い、**再学習の前に**似た候補を棄却する。この違いは明記すること
- **γ の扱い**：NEAS の式では γ は品質にかかる定数倍だが、同じ個数の組み合わせを比べる限り det は γ^{2K} 倍されるだけで選ばれる組み合わせが変わらない。そこで本実装では指数（r^γ）とし、精度と多様性の重みを調整できるようにした
- **類似度の特徴**：NEAS は各パスの出力特徴マップ、本実装は最終層直前の 1280 次元特徴を使う。β は中央値ヒューリスティックで自動設定
- **品質の評価データ**：学習データのうち、検証用とは別の 2,000 枚（NEAS の ACC_train′ に相当）。検証データはアンサンブルの組み合わせ決定用に使わずに残す
- **BN 再計算**：継承重みで評価する前に、各候補の BN 統計を学習データで再計算する
- **選択アルゴリズム**：det を最大化する部分集合は貪欲法で近似（DPP の MAP 推定 [6, 8]）

### 6.2 コスト比較で報告すべき指標
- 探索時間は 1 回分（約 65 分）で、K 個取り出してもコストは増えない
- 再学習は 1 モデル約 45 分なので、K 個なら約 45K 分。総コストは探索 + 再学習で比べる
- DPP による選択そのものは候補 12 個で約 16 秒（再学習に比べて無視できる）
- 精度は **総 FLOPs・総パラメータ数をそろえた条件**で、単体モデルと比較する
- 比較対象：(a) 探索で得た単体の最良構造、(b) seed を変えて K 回探索した構造のアンサンブル、(c) `topk` / `order` で選んだ構造のアンサンブル

### 6.3 多様性の指標
- 予測の食い違い率 [13] と oracle 精度は `cifar10_ensemble_eval.py` が出力する
- アンサンブルの NLL も報告すると、較正の観点からの比較ができる [10]

## 7. 引用文献

書誌情報は出版社のページで最終確認すること。

| # | 文献 | 引用する箇所 |
|---|---|---|
| [1] | H. Cai, L. Zhu, S. Han. "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware." ICLR 2019. arXiv:1812.00332 | ベースの探索手法 |
| [2] | A. Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." Technical Report, University of Toronto, 2009. | CIFAR-10 |
| [3] | M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018. arXiv:1801.04381 | 探索空間の MBConv ブロック |
| [4] | M. Chen, H. Peng, J. Fu, H. Ling. "One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking." CVPR 2021. arXiv:2104.00597 | Diversity Score（Eq.4, 5, 7, 8）、BN 再計算 |
| [5] | S. Zaidi, A. Zela, T. Elsken, C. Holmes, F. Hutter, Y. W. Teh. "Neural Ensemble Search for Uncertainty Estimation and Dataset Shift." NeurIPS 2021. arXiv:2006.08573 | アンサンブル向け NAS の先行研究 |
| [6] | A. Kulesza, B. Taskar. "Determinantal Point Processes for Machine Learning." Foundations and Trends in Machine Learning, 5(2–3), 2012. | DPP カーネル、MAP 推定 |
| [7] | A. Kulesza, B. Taskar. "k-DPPs: Fixed-Size Determinantal Point Processes." ICML 2011. | 個数を固定した DPP（NEAS が依拠） |
| [8] | L. Chen, G. Zhang, H. Zhou. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." NeurIPS 2018. arXiv:1709.05135 | 貪欲法による MAP 推定（本実装は同じ目的関数の素朴な貪欲法） |
| [9] | Z. Guo, X. Zhang, H. Mu, W. Heng, Z. Liu, Y. Wei, J. Sun. "Single Path One-Shot Neural Architecture Search with Uniform Sampling." ECCV 2020. arXiv:1904.00420 | 継承重みで評価する前の BN 再計算 |
| [10] | B. Lakshminarayanan, A. Pritzel, C. Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017. arXiv:1612.01474 | softmax 平均によるアンサンブル、NLL による評価 |
| [11] | I. Loshchilov, F. Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. arXiv:1608.03983 | cosine 学習率スケジュール |
| [12] | C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016. | label smoothing |
| [13] | L. I. Kuncheva, C. J. Whitaker. "Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy." Machine Learning, 51(2), 2003. | 予測の食い違い率（disagreement） |
| [14] | R. Caruana, A. Niculescu-Mizil, G. Crew, A. Ksikes. "Ensemble Selection from Libraries of Models." ICML 2004. | 訓練済みモデルからの前向き選択（NES [5] が使用）。本研究の「再学習前の選択」との対比 |
