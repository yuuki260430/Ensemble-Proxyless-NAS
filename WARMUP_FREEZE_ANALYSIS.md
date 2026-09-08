# Warmup Epoch フリーズ問題の分析

## 問題
`search/cifar10_arch_search.py`実行時、`Warmup epoch: 1`が表示された直後にプログラムが停止

### ログ出力状況
```
Total training params: 14.77M
Total FLOPs: 5.2M

 ------------------------------ Warmup epoch: 1 ------------------------------ 
```
→ この後、追加出力なし（プロセス停止）

---

## 実行フロー分析

```
cifar10_arch_search.py (行207-209)
    ↓
    arch_search_run_manager.warm_up(warmup_epochs=20)
        ↓
        nas_manager.py::warm_up() (行247-330)
            ↓
            254行：print('Warmup epoch: 1')  ← ✓ ここまで実行
            ↓
            255-259行：AverageMeterの初期化
            ↓
            260行：self.run_manager.net.train()
            ↓
            262行：end = time.time()
            ↓
            266行：for i, (images, labels) in enumerate(data_loader):  ← ❌ ここで停止？
                  ↓
                  [データイテレーションがハング]
```

---

## 最も可能性の高い停止原因

### 1. **マルチプロセッシング初期化でのハング** ⭐ 最有力

**場所**: nas_manager.py::266行の`enumerate(data_loader)`

**理由**:
- CIFAR-10 DataLoaderは `num_workers=4` に設定（マルチプロセッシング）
- forループでの最初のデータ取得時にワーカープロセスが初期化される
- GPU環境でのマルチプロセッシング初期化でのハング可能性

**設定内容** (search/data_providers/cifar10.py):
```python
self.train = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=256, 
    sampler=train_sampler,
    num_workers=4,          # ← マルチプロセッシング
    pin_memory=True,        # ← GPU転送最適化
    drop_last=True,
)
```

### 2. 候補: DataLoaderの長さ計算でハング

**場所**: nas_manager.py::251行の`nBatch = len(data_loader)`

DataLoaderの長さ計算に時間がかかる可能性

### 3. 候補: ネットワークのデバイス移動

**場所**: nas_manager.py::260行の`self.run_manager.net.train()`

ネットワークがGPUに存在しない、または初期化エラー

---

## 処理詳細（各ステップ）

### Step 1: Warmup準備 ✓
```python
# 行247-252
def warm_up(self, warmup_epochs=25):
    lr_max = 0.05
    data_loader = self.run_manager.run_config.train_loader  # 行249
    nBatch = len(data_loader)                               # 行251
    T_total = warmup_epochs * nBatch
```

### Step 2: Epoch開始 ✓
```python
# 行253-265
for epoch in range(self.warmup_epoch, warmup_epochs):
    print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
    # ← ここまで正常に実行される
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # ... AverageMeter初期化
    
    self.run_manager.net.train()  # 行260
    end = time.time()             # 行262
```

### Step 3: データローディング ❌ **ここで停止**
```python
# 行266
for i, (images, labels) in enumerate(data_loader):
    # ← ここでハング
    # マルチプロセッシング（num_workers=4）初期化待機中？
    
    data_time.update(time.time() - end)  # 行267
    # ... 以下のバッチ処理は実行されない
```

### Step 4: バッチ処理（実行されない）
```python
# 行268-295
T_cur = epoch * nBatch + i
warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
# ... 勾配計算、backward, optimizer.step()
```

### Step 5: 検証（実行されない）
```python
# 行320
valid_res, flops, latency = self.validate()
```

---

## 実行時パラメータ確認

**cifar10_arch_search.pyの実行設定**:
- dataset: cifar10
- train_batch_size: 256
- n_worker: 4 ← マルチプロセッシング有効
- resize_scale: 0.08
- distort_color: normal
- warmup_epochs: 20 ← デフォルト値

**CIFAR-10データ設定**:
```
データセット: torchvision.datasets.CIFAR10
保存場所: ./data/cifar10/
トレーニングサンプル: 50,000
テストサンプル: 10,000
画像サイズ: 32x32
クラス数: 10
```

---

## 根本原因の特定方法

### 方法1: num_workersを0に変更
マルチプロセッシングを無効化して検証

**修正ファイル**: [search/data_providers/cifar10.py](search/data_providers/cifar10.py#L42)

```python
self.train = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
    num_workers=0,  # ← 4から0に変更
    pin_memory=True,
    drop_last=True,
)
```

### 方法2: マルチプロセッシング安全化
メインスクリプトで`if __name__ == '__main__':`を使用

**修正ファイル**: [search/cifar10_arch_search.py](search/cifar10_arch_search.py)

最後に以下を追加:
```python
if __name__ == '__main__':
    # 既存のmain処理をここに移動
    # 実際のコードをここに配置
```

### 方法3: pin_memory無効化（GPU関連エラーの場合）

```python
pin_memory=False,  # ← Trueから変更
```

---

## 予想される停止時間

マルチプロセッシング初期化でハングしている場合：
- **短期**: 数秒～数十秒（初期化待機）
- **長期**: 数分以上（デッドロック状態）
- **実際**: output.logが作成されたが更新されない状態

---

## 推奨される次のステップ

1. **即座の検証**:
   - `num_workers=0`に設定して実行
   - 上記で成功したら、マルチプロセッシング関連の問題と確定

2. **原因の詳細化**:
   - PyTorchバージョン確認
   - GPU環境でのマルチプロセッシング互換性確認
   - システムリソース確認（メモリ、プロセス数制限）

3. **長期的な解決**:
   - マルチプロセッシング安全化
   - 環境に応じたnum_workers値の最適化
   - エラーハンドリングの追加

---

## 関連コード参照

| ファイル | 行番号 | 処理内容 |
|---------|--------|---------|
| [search/cifar10_arch_search.py](search/cifar10_arch_search.py#L207) | 207-209 | `warm_up()`呼び出し |
| [search/nas_manager.py](search/nas_manager.py#L247) | 247-330 | `warm_up()`実装 |
| [search/run_manager.py](search/run_manager.py#L103) | 103-106 | `train_loader`プロパティ |
| [search/data_providers/cifar10.py](search/data_providers/cifar10.py#L42) | 42-46 | DataLoader設定 |
| [search/data_providers/base_provider.py](search/data_providers/base_provider.py) | 1-50 | DataProvider基本 |
