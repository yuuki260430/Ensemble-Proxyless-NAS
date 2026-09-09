# 環境別YAML設定の実装完了報告

## 実装概要

A6000環境とRTX 5080環境の両方で実行可能なように、ハードウェア別の設定ファイルシステムを実装しました。

## 主な変更内容

### 1. 新規ファイル

| ファイル | 説明 |
|---------|------|
| `search/configs/config_a6000.yaml` | A6000環境用ハードウェア設定 |
| `search/configs/config_5080.yaml` | RTX 5080環境用ハードウェア設定 |
| `search/utils/config_loader.py` | YAML設定ファイルの読み込みユーティリティ |
| `search/configs/README.md` | 設定ファイルの使用方法ドキュメント |

### 2. 修正ファイル

| ファイル | 修正内容 |
|---------|---------|
| `search/run_manager.py` | `latency_estimator_url`パラメータを追加 |
| `search/nas_manager.py` | `ArchSearchRunManager`に環境別URL対応 |
| `search/utils/latency_estimator.py` | URLがNoneの場合に空辞書を使用 |
| `search/utils/__init__.py` | インポートを相対パスに修正 |
| `search/cifar10_arch_search.py` | `--hw-config`オプション追加、設定読み込み実装 |
| `search/imagenet_arch_search.py` | `--hw-config`オプション追加、設定読み込み実装 |
| `search/cifar10_run_exp.py` | `--hw-config`オプション追加、設定読み込み実装 |
| `search/imagenet_run_exp.py` | `--hw-config`オプション追加、設定読み込み実装 |

## 使用方法

### 基本的な実行コマンド

```bash
# RTX 5080でCIFAR-10アーキテクチャ探索
python search/cifar10_arch_search.py \\
  --path Exp2/proxyless_gpu \\
  --dropout 0.3 \\
  --hw-config search/configs/config_5080.yaml

# A6000でImageNetアーキテクチャ探索
python search/imagenet_arch_search.py \\
  --path Exp/proxyless_gpu \\
  --hw-config search/configs/config_a6000.yaml
```

### 訓練スクリプト

```bash
# RTX 5080でCIFAR-10訓練
python search/cifar10_run_exp.py \\
  --path Exp/cifar10 \\
  --train \\
  --net proxyless_mobile \\
  --hw-config search/configs/config_5080.yaml

# A6000でImageNet訓練
python search/imagenet_run_exp.py \\
  --path Exp/imagenet \\
  --train \\
  --net proxyless_gpu \\
  --hw-config search/configs/config_a6000.yaml
```

## 設定ファイルの仕様

各設定ファイルはYAML形式で、以下の構造を持ちます：

```yaml
# ハードウェア情報
hardware:
  name: デバイス名
  gpu_memory: メモリ容量(GB)
  architecture: アーキテクチャ
  compute_capability: コンピュート能力
  max_power: 最大電力(W)

# レイテンシモデルの設定
mobile_latency_model: "モバイル用YAML URL"
gpu_a6000_latency_model: "A6000用YAML URL"
cpu_latency_model: "CPU用YAML URL"
gpu8_latency_model: "GPU8用YAML URL"

# その他の設定
cache_dir: "キャッシュディレクトリ"
overwrite_cache: キャッシュ上書きフラグ
notes: "環境について注釈"
```

## 実装の特徴

1. **柔軟性**: `--hw-config`オプションで任意の設定ファイルを指定可能
2. **後方互換性**: オプション指定がない場合はデフォルト動作
3. **環境変数対応**: `PROXYLESS_ENV`で自動選択可能（オプション）
4. **拡張性**: 新しいハードウェアの設定ファイルを簡単に追加可能
5. **文書化**: 詳細な使用方法ドキュメントを提供

## トラブルシューティング

### YAMLファイルが見つからない

```
FileNotFoundError: Config file not found: search/configs/config_xxxx.yaml
```

**解決方法**:
- ファイルパスが正しいか確認
- ファイルが`search/configs/`ディレクトリに存在するか確認

### モジュールのインポートエラー

```
ModuleNotFoundError: No module named 'torchvision'
```

**解決方法**:
```bash
pip install torchvision torch pyyaml
```

### メモリ不足エラー

RTX 5080（メモリ14GB）ではA6000（メモリ48GB）より容量が少ないため、必要に応じてパラメータを調整してください：

```bash
python search/cifar10_arch_search.py \\
  --path Exp2/proxyless_gpu \\
  --dropout 0.3 \\
  --train_batch_size 128 \\
  --n_worker 2 \\
  --hw-config search/configs/config_5080.yaml
```

## Git管理について

設定ファイル（YAML）はGit管理下に置かれています。環境ごとの微調整が必要な場合は：

1. 既存の設定ファイルをコピー
2. 必要な部分を修正
3. 新しい名前で保存（例：`config_custom.yaml`）
4. `--hw-config`で指定

## 今後の拡張

1. **ハードウェア自動検出**: GPUタイプから設定ファイルを自動選択
2. **動的パラメータ調整**: メモリに基づいてバッチサイズを自動調整
3. **パフォーマンスプロファイリング**: ハードウェア固有のレイテンシモデル生成
4. **複数GPU対応**: マルチGPU環境での自動最適化
