# ProxylessNAS 環境別設定ファイル

## 概要

このプロジェクトは、A6000とRTX 5080などの異なるハードウェア環境で実行されることを想定しています。各環境用の設定ファイルは以下のようになっています：

- `config_a6000.yaml`: NVIDIA A6000 GPU環境用
- `config_5080.yaml`: NVIDIA RTX 5080 GPU環境用

## 環境別の実行方法

### 基本的な使用方法

すべてのスクリプト（`cifar10_arch_search.py`, `imagenet_arch_search.py`, `cifar10_run_exp.py`, `imagenet_run_exp.py`）は、`--hw-config` オプションで環境別設定ファイルを指定できます。

```bash
# CIFAR-10アーキテクチャ探索 (RTX 5080環境)
python search/cifar10_arch_search.py --path Exp2/proxyless_gpu --dropout 0.3 --hw-config search/configs/config_5080.yaml

# ImageNetアーキテクチャ探索 (A6000環境)
python search/imagenet_arch_search.py --path Exp/proxyless_gpu --hw-config search/configs/config_a6000.yaml

# CIFAR-10訓練 (RTX 5080環境)
python search/cifar10_run_exp.py --path Exp/cifar10 --train --net proxyless_mobile --hw-config search/configs/config_5080.yaml

# ImageNet訓練 (A6000環境)
python search/imagenet_run_exp.py --path Exp/imagenet --train --net proxyless_gpu --hw-config search/configs/config_a6000.yaml
```

### 環境変数を使用した自動判定

`PROXYLESS_ENV` 環境変数を設定することで、デフォルト設定ファイルを自動選択できます（オプション指定がない場合）：

```bash
# RTX 5080用のデフォルト設定を使用
export PROXYLESS_ENV=5080
python search/cifar10_arch_search.py --path Exp2/proxyless_gpu --dropout 0.3

# A6000用のデフォルト設定を使用
export PROXYLESS_ENV=a6000
python search/imagenet_arch_search.py --path Exp/proxyless_gpu
```

## 設定ファイルの構造

各設定ファイル（YAML形式）には以下の情報が含まれます：

```yaml
# ハードウェア情報
hardware:
  name: "デバイス名"
  gpu_memory: メモリ容量(GB)
  architecture: "アーキテクチャ名"
  compute_capability: "コンピュート能力"
  max_power: "最大電力(W)"

# レイテンシモデルURL
mobile_latency_model: "モバイルデバイス用YAMLのURL"
gpu_xxxx_latency_model: "GPU固有のレイテンシモデル"
cpu_latency_model: "CPU用レイテンシモデル"

# その他の設定
cache_dir: "キャッシュディレクトリ"
overwrite_cache: キャッシュ上書きフラグ
```

## カスタムレイテンシモデルの設定

ハードウェア固有のレイテンシモデルYAMLを使用する場合：

1. YAMLファイルを準備
2. 設定ファイルの対応するキーにURLまたはパスを指定
3. スクリプト実行時に `--hw-config` でその設定ファイルを指定

例：

```yaml
# config_custom.yaml
gpu_a6000_latency_model: "https://example.com/a6000_latency.yaml"
# または
gpu_a6000_latency_model: "/path/to/local/latency_model.yaml"
```

## 設定ファイルの作成

新しいハードウェア用の設定ファイルを作成する場合は、既存のファイル（`config_a6000.yaml` または `config_5080.yaml`）をテンプレートとして使用してください。

## トラブルシューティング

### YAMLファイルが見つからない

```
FileNotFoundError: Config file not found: search/configs/config_xxxx.yaml
```

**解決方法**：
- ファイルパスが正しいか確認
- ファイルがGit管理下にあるか確認

### レイテンシモデルURLが無効

YAMLダウンロード時にエラーが発生する場合：

1. URLが正しいか確認
2. インターネット接続を確認
3. `overwrite_cache: true` でキャッシュを強制更新

### メモリ不足エラー

ハードウェアのメモリ容量がA6000からRTX 5080に減った場合：

1. `--train_batch_size` を減らす
2. `--n_worker` を減らす
3. `--valid_size` を減らす

例：
```bash
python search/cifar10_arch_search.py \
  --path Exp2/proxyless_gpu \
  --dropout 0.3 \
  --train_batch_size 128 \
  --hw-config search/configs/config_5080.yaml
```

## 依存パッケージ

実行前に以下のパッケージをインストールしてください：

```bash
pip install pyyaml torch torchvision numpy
```

より詳細な環境セットアップについては、プロジェクトのメインREADMEを参照してください。
