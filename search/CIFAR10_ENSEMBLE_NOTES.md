# CIFAR-10 化とアンサンブル探索：作業記録と論文用メモ

最終更新: 2026-09-30

ProxylessNAS を CIFAR-10 / RTX 5080 で動かすための変更、アンサンブル用に追加した機能、supernet 学習の調査と改善、実験結果、論文に書くべき引用と手法上の注意点をまとめたもの。

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
- 既知の問題：1 回だけ `libnvrtc.so.12` 内でセグメンテーション違反が発生（試運転 7 回中 1 回、以降の長時間実行では発生なし）。`cudnn.benchmark=True`（[run_manager.py](run_manager.py)）時の実行時カーネルコンパイルが疑わしい。再発したら `--resume` で再開し、頻発するなら benchmark 無効化か cu130 版を試す。

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
# 探索（fair ウォームアップ付き。4 章参照）
python search/cifar10_arch_search.py --path Exp3/search --gpu 0 --warmup_epochs 70 --warmup_fair \
  --n_epochs 150 --dropout 0.3 --hw-config search/configs/config_5080.yaml
# 構造の取り出し
python search/cifar10_derive_archs.py --search_path Exp3/search --out_path Exp3/ensemble_dpp \
  --n_archs 3 --mode sample --select dpp --pool_size 30
# 各構造の再学習
for d in Exp3/ensemble_dpp/arch_*; do
  python search/cifar10_run_exp.py --path $d --train --gpu 0 --hw-config search/configs/config_5080.yaml > $d/train.log 2>&1
done
# 評価
python search/cifar10_ensemble_eval.py --paths Exp3/ensemble_dpp/arch_* --output Exp3/ensemble_dpp/ensemble_test.json
```

注意：`cifar10_run_exp.py` の `--path` には `net.config` があるフォルダを指定すること。探索の出力フォルダ（例：`Exp2/proxyless_gpu`）を指定すると、ImageNet 用の構造と重みが読み込まれて size mismatch になる。

## 4. supernet 学習の調査と改善

### 4.1 症状
最初の探索（5.1 節）で、supernet の精度がウォームアップ 20 エポックで 20%、探索終了時でも 58% にとどまった。同じ構造を固定ネットワークとして学習すると 89% に達する。

### 4.2 見つかった実装上のバグ（コミット `1e7eaa4` で修正）
どちらも、最初に取り込んだ元のコード（`a195216`）から変更していない部分にあった。

1. **探索フェーズで候補演算の重みが学習されない**（[modules/mix_op.py](modules/mix_op.py) の `ArchGradientFunction`）
   - `full_v2` モードの backward が `torch.autograd.grad(output, detached_x)` で入力の勾配だけを返しており、候補演算の重みに勾配が届いていなかった（0/162 パラメータ）。
   - そのため最初の探索では、探索フェーズ 150 エポックの間、stem・最初の固定ブロック・feature_mix・分類層と α だけが学習され、候補演算はウォームアップ時点の重みのままだった。
   - `torch.utils.checkpoint` と同じ方式（backward 内で `torch.autograd.backward`）に修正。通常の自動微分と勾配が一致することを確認（194/194 パラメータ、float64 で相対誤差 2.5e-16）。
2. **選ばれなかった候補演算に古い勾配が残る**（[nas_manager.py](nas_manager.py)）
   - `net.zero_grad()` を、使わない候補を `unused_modules_off()` でモデルから外した後に呼んでいたため、外れた演算の勾配が消えず、`optimizer.step()` で繰り返し適用されていた。
   - 更新前に `optimizer.zero_grad()` を追加。効果は小さかった（ウォームアップ 1 エポックの学習精度 10.7% → 11.9%）。

### 4.3 立ち上がりの遅さの切り分け
最小の学習ループ（1 エポック = 174 ステップ、学習精度はステップ 100〜174 の平均）で比較した。

| 条件 | 学習 top-1 |
|---|---|
| 固定ネットワーク、ランダム初期値から（参考：3 エポックで検証 64%） | 1 エポックで検証 46% |
| supernet・経路を固定 | 41% |
| supernet・2 経路を交互 | 31% |
| supernet・毎ステップ一様ランダム（元の方式） | 11% |
| 同上で lr 1/10 / モーメンタムなし / Zero 除外 | 10〜12% |

- 経路の切り替えの仕組みは正常。経路の選択肢が増えるほど立ち上がりが遅くなる。各層の候補は重みを共有しない別々の演算なので、下流の層は上流の 6〜7 通りの出力すべてに対応する必要がある。
- ImageNet（1 エポック約 5,000 ステップ、元論文のウォームアップ 40 エポックで約 20 万ステップ）と比べ、CIFAR-10（1 エポック約 175 ステップ、ウォームアップ 20 エポックで約 3,500 ステップ）は**更新ステップ数が約 1/57** しかない。ウォームアップをエポック数で決めると、CIFAR-10 ではほとんど学習されないまま探索が始まる。

### 4.4 ウォームアップを延ばした場合（`Exp3/warmup150`）
一様ランダムのまま 150 エポック（約 26,000 ステップ、約 55 分）に延ばすと、検証 46.1% / 学習 50.5% で頭打ちになった。学習精度が低いので過学習ではなく、当てはめきれていない。ステップ数を増やすだけでは不十分。

### 4.5 FairNAS 型のウォームアップ（`--warmup_fair`、コミット `a0a6b4f`）
FairNAS [15] の strict fairness を参考に、1 回の更新ごとに、各層の全候補を 1 回ずつ使う m 本の経路（m = 7）で勾配を貯めてから更新する。

- `fair_paths()`：全 21 層の全候補を 1 回ずつ含む 7 本の経路を作る。候補が 6 個の層（21 層中 6 層）では、残り 1 本をランダムに選ぶ
- `average_fair_grads()`：各パラメータの勾配を、そのパラメータが使われた回数で割る（共有部分は 7 本の平均、候補演算は通常 1 本分）
- 1 エポックの学習時間は約 78 秒（一様ランダムの約 6.5 倍）

**比較結果（ほぼ同じ計算時間）**

| | 所要時間 | 検証 top-1 | 学習 top-1 |
|---|---|---|---|
| 一様ランダム 150 エポック（`Exp3/warmup150`） | 約 55 分 | 46.1% | 50.5% |
| **fair 35 エポック（`Exp3/fair35`）** | **約 46 分** | **62.4%** | **64.8%** |

- fair は 10 エポック（約 13 分）で 46.7% に達し、一様ランダム 55 分の最終値を約 1/4 の時間で上回った
- 学習精度 64.8% なので、まだ当てはめきれていない。エポック数を増やせば伸びる余地がある

## 5. 実験結果

### 5.1 最初のアーキテクチャ探索（`Exp2/proxyless_gpu`）※4.2 節のバグの影響あり
- 設定：ウォームアップ 20 エポック + 探索 150 エポック、batch 256、lr 0.05（cosine）、dropout 0.3、`--target_hardware None`、勾配法（`full_v2`）
- 所要時間：約 65 分（ウォームアップ約 8 分 + 探索約 57 分、ログの時刻と 1 反復あたりの時間から推定）

| 時点 | supernet 検証 top-1 |
|---|---|
| ウォームアップ 20/20 | 20.5% |
| 探索 150/150 | 58.3%（最高 58.6%） |

- 探索終了時の α のエントロピーは 1 ブロックあたり平均 1.62（全 21 ブロック合計 34.1）。一様分布なら平均 1.90（候補 7 個のブロックは ln 7、6 個のブロックは ln 6）なので、確率分布はまだほぼ一様
- 得られた構造：163.0M FLOPs、3.28M パラメータ
- **候補演算の重みがほぼ学習されていない supernet 上で α を更新しているため、この探索結果の構造選択には信頼性がない。アンサンブル実験には使わず、修正後のコードで探索し直す**

### 5.2 5.1 の構造の再学習（`Exp2/cifar10_retrain`）
- 設定：`learned_net/run.config` を使用（150 エポック、lr 0.05、label smoothing 0.1、wd 4e-5）。supernet の継承重みから学習開始
- 所要時間：約 45 分

| | top-1 |
|---|---|
| 学習前（継承重みのまま） | テスト 59.4% |
| 再学習後（150 エポック） | **検証 89.16%（最高 89.46%）/ テスト 89.10%** |

→ 暫定の単体ベースライン：テスト 89.10%、163.0M FLOPs、3.28M パラメータ（構造は 5.1 のバグありの探索で選ばれたもの）

### 5.3 構造の取り出し（5.1 の supernet での動作確認のみ、再学習は未実施）
- `topk`：最も確率の高い構造から 1〜2 ブロックしか違わない。`--min_hamming 4` で 4 ブロック以上違う組み合わせにできる
- `sample`：最も確率の高い構造から 12〜16 ブロック違い、ほぼランダム
- `dpp`（候補 12 個から 3 個）：選ばれた 3 構造の類似度は 0.37〜0.39（候補全体では 0.32〜0.60）。類似度が最も低い候補（0.32）は品質 56% と低いため選ばれず、品質と多様性の両方が考慮されている
- 継承重みのまま 3 構造をアンサンブルしても、テスト精度 59.0%（単体 59.4%）で向上しない。重みを共有しているため誤りも似ていると考えられ、効果は再学習後に評価する必要がある

## 6. 未解決の課題と今後の予定

1. fair ウォームアップを延ばす（例：70 エポック）と、修正後の探索フェーズ（150 エポック）を通して回し、α が収束するか確認する（最初の探索ではエントロピー 1.62 とほぼ一様）
2. 修正後の探索結果で、`topk` / `order` と `dpp` の選び方を再学習込みで比較する
3. 総 FLOPs をそろえた単体モデルとの比較、seed を変えた複数回探索との比較
4. supernet がなお当てはめきれていない場合は、段階的に候補を増やす方式（Once-for-All [16] の progressive shrinking の考え方）を検討する
5. `cifar10_run_exp.py --net proxyless_*`（`net.config` がないフォルダを指定した場合）は ImageNet 用の公式構造を読み込む。探索結果を使う通常の流れには影響しない
6. ログの `MixedEdge do not support set_arch_param_grad()` は、Zero が選ばれた回に `AP_path_wb.grad` が空のため出るもので、元の実装と同じ挙動（実害なし）

## 7. 論文に書くべき点

### 7.1 元の手法からの変更（手法・実験設定の節に明記）
- **CIFAR-10 向けの supernet 変更**：stem stride 2→1、第 1 段 stride 2→1（最終特徴マップ 4×4）
- **ハードウェア制約**：mobile レイテンシ表は 224 入力専用なので FLOPs を使用。基準値は公式構造の FLOPs 比（0.28〜0.30）から ImageNet の値を換算
- **実装の修正**：公開実装の `ArchGradientFunction` は、現在の PyTorch では探索フェーズで候補演算の重みに勾配を渡さない。修正して実験したことを再現性のために明記する（元論文当時の環境での挙動は未確認なので、元論文の結果に対する主張はしない）
- **FairNAS 型ウォームアップ**：FairNAS [15] の strict fairness をウォームアップに適用。次の 2 点は FairNAS からの変更として明記する
  - 勾配を、そのパラメータが使われた回数で割る（FairNAS は単純な和。この設定では和にすると最初の 10 ステップで損失が平均 22 まで発散した）
  - 候補数が層によって 6 / 7 と異なるため、6 候補の層では 1 候補だけ 2 回使う
- **ウォームアップの長さ**：エポック数ではなく更新ステップ数で ImageNet 設定と比較して決めたことを書く（CIFAR-10 は 1 エポックのステップ数が約 1/29）
- **NEAS の Diversity Score の転用**：NEAS [4] は層ごとの演算の組み合わせを評価して、supernet 学習中に探索空間を縮める。本研究では同じ DPP カーネルを**構造単位**で使い、**再学習の前に**似た候補を棄却する。この違いは明記すること
- **γ の扱い**：NEAS の式では γ は品質にかかる定数倍だが、同じ個数の組み合わせを比べる限り det は γ^{2K} 倍されるだけで選ばれる組み合わせが変わらない。そこで本実装では指数（r^γ）とし、精度と多様性の重みを調整できるようにした
- **類似度の特徴**：NEAS は各パスの出力特徴マップ、本実装は最終層直前の 1280 次元特徴を使う。β は中央値ヒューリスティックで自動設定
- **品質の評価データ**：学習データのうち、検証用とは別の 2,000 枚（NEAS の ACC_train′ に相当）。検証データはアンサンブルの組み合わせ決定用に使わずに残す
- **BN 再計算**：継承重みで評価する前に、各候補の BN 統計を学習データで再計算する
- **選択アルゴリズム**：det を最大化する部分集合は貪欲法で近似（DPP の MAP 推定 [6, 8]）

### 7.2 コスト比較で報告すべき指標
- 探索時間は 1 回分で、K 個取り出してもコストは増えない（fair ウォームアップ込みの時間を報告する）
- 再学習は 1 モデル約 45 分なので、K 個なら約 45K 分。総コストは探索 + 再学習で比べる
- DPP による選択そのものは候補 12 個で約 16 秒（再学習に比べて無視できる）
- 精度は **総 FLOPs・総パラメータ数をそろえた条件**で、単体モデルと比較する
- 比較対象：(a) 探索で得た単体の最良構造、(b) seed を変えて K 回探索した構造のアンサンブル、(c) `topk` / `order` で選んだ構造のアンサンブル
- supernet 学習の比較（一様ランダム vs fair）は、エポック数ではなく計算時間をそろえて示す（4.5 節）

### 7.3 多様性の指標
- 予測の食い違い率 [13] と oracle 精度は `cifar10_ensemble_eval.py` が出力する
- アンサンブルの NLL も報告すると、較正の観点からの比較ができる [10]

## 8. 引用文献

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
| [9] | Z. Guo, X. Zhang, H. Mu, W. Heng, Z. Liu, Y. Wei, J. Sun. "Single Path One-Shot Neural Architecture Search with Uniform Sampling." ECCV 2020. arXiv:1904.00420 | 一様サンプリングによる supernet 学習、継承重みで評価する前の BN 再計算 |
| [10] | B. Lakshminarayanan, A. Pritzel, C. Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017. arXiv:1612.01474 | softmax 平均によるアンサンブル、NLL による評価 |
| [11] | I. Loshchilov, F. Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. arXiv:1608.03983 | cosine 学習率スケジュール |
| [12] | C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016. | label smoothing |
| [13] | L. I. Kuncheva, C. J. Whitaker. "Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy." Machine Learning, 51(2), 2003. | 予測の食い違い率（disagreement） |
| [14] | R. Caruana, A. Niculescu-Mizil, G. Crew, A. Ksikes. "Ensemble Selection from Libraries of Models." ICML 2004. | 訓練済みモデルからの前向き選択（NES [5] が使用）。本研究の「再学習前の選択」との対比 |
| [15] | X. Chu, B. Zhang, R. Xu. "FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search." ICCV 2021. arXiv:1907.01845 | strict fairness による supernet 学習（`--warmup_fair`） |
| [16] | H. Cai, C. Gan, T. Wang, Z. Zhang, S. Han. "Once-for-All: Train One Network and Specialize it for Efficient Deployment." ICLR 2020. arXiv:1908.09791 | progressive shrinking（今後の検討候補） |
