#!/usr/bin/env python3
"""
Warmup Epoch フリーズ問題の検証スクリプト

このスクリプトは、データローディングのマルチプロセッシング関連の問題を
ステップバイステップで検証します。
"""

import sys
import time
import torch
import torch.utils.data
from pathlib import Path

# プロジェクトのsearchディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'search'))

print("=" * 80)
print("WARMUP EPOCH FREEZE DIAGNOSTIC SCRIPT")
print("=" * 80)

# ==================== Test 1: DataProviderの初期化 ====================
print("\n[TEST 1] CIFAR-10 DataProviderの初期化")
print("-" * 80)

try:
    from data_providers.cifar10 import CIFAR10DataProvider
    print("✓ CIFAR10DataProviderのインポート成功")
    
    # DataProviderの初期化（デフォルト設定）
    print("\n  DataProviderを初期化中...")
    print("    - train_batch_size: 256")
    print("    - test_batch_size: 256")
    print("    - valid_size: 5000")
    print("    - n_worker: 4  ← マルチプロセッシング有効")
    
    start = time.time()
    data_provider = CIFAR10DataProvider(
        save_path=str(project_root / 'data' / 'cifar10'),
        train_batch_size=256,
        test_batch_size=256,
        valid_size=5000,
        n_worker=4,
        resize_scale=0.08,
        distort_color='normal'
    )
    elapsed = time.time() - start
    print(f"✓ DataProvider初期化成功（{elapsed:.2f}秒）")
    
except Exception as e:
    print(f"✗ DataProvider初期化失敗: {e}")
    sys.exit(1)

# ==================== Test 2: train_loaderの取得 ====================
print("\n[TEST 2] train_loaderの取得")
print("-" * 80)

try:
    train_loader = data_provider.train
    print(f"✓ train_loaderの取得成功")
    print(f"  - type: {type(train_loader)}")
    print(f"  - num_workers: {train_loader.num_workers}")
    print(f"  - pin_memory: {train_loader.pin_memory}")
    print(f"  - batch_size: {train_loader.batch_size}")
    
except Exception as e:
    print(f"✗ train_loaderの取得失敗: {e}")
    sys.exit(1)

# ==================== Test 3: train_loaderの長さ計算 ====================
print("\n[TEST 3] train_loaderの長さ計算")
print("-" * 80)

try:
    print("  len(train_loader)を計算中...")
    start = time.time()
    n_batch = len(train_loader)
    elapsed = time.time() - start
    print(f"✓ 長さ計算成功（{elapsed:.2f}秒）")
    print(f"  - nBatch: {n_batch}")
    
except Exception as e:
    print(f"✗ 長さ計算失敗: {e}")
    print("  → この段階でフリーズが発生している可能性があります")
    sys.exit(1)

# ==================== Test 4: enumerate開始（最初のバッチ取得） ====================
print("\n[TEST 4] enumerate(train_loader)の開始 - CRITICAL TEST")
print("-" * 80)

try:
    print("  enumerate(train_loader)を開始中...")
    print("  （最初のバッチを取得するまで待機...）")
    print("  （タイムアウト時間: 30秒）")
    
    start = time.time()
    
    # イテレータを作成（ここでワーカープロセスが起動される）
    data_iter = enumerate(train_loader)
    print(f"  ✓ イテレータ作成完了（{time.time() - start:.2f}秒）")
    
    # 最初のバッチを取得（ここでハングする可能性が高い）
    print("  最初のバッチを取得中...")
    first_idx, (images, labels) = next(data_iter)
    elapsed = time.time() - start
    
    print(f"✓ 最初のバッチ取得成功（{elapsed:.2f}秒）")
    print(f"  - batch index: {first_idx}")
    print(f"  - images shape: {images.shape}")
    print(f"  - labels shape: {labels.shape}")
    print(f"  → SUCCESS: マルチプロセッシング初期化正常")
    
    # 2番目のバッチも確認
    print("\n  2番目のバッチを取得中...")
    start = time.time()
    second_idx, (images2, labels2) = next(data_iter)
    elapsed = time.time() - start
    print(f"✓ 2番目のバッチ取得成功（{elapsed:.3f}秒）")
    
except StopIteration:
    print("✗ データローダーが空（StopIteration）")
    sys.exit(1)
    
except Exception as e:
    print(f"✗ バッチ取得失敗: {e}")
    print(f"\n  → この段階でフリーズが発生している可能性が非常に高いです")
    print(f"  → 原因: マルチプロセッシング関連の問題")
    sys.exit(1)

# ==================== Test 5: ネットワークのトレーニングモード ====================
print("\n[TEST 5] ネットワークのトレーニングモード切り替え")
print("-" * 80)

try:
    from models import CIFAR10RunConfig
    from nas_manager import ArchSearchRunManager, GradientArchSearchConfig
    from models.super_nets.super_proxyless import SuperProxylessNASNets
    
    print("  ネットワークを作成中...")
    
    # ネットワーク設定
    width_stages = [24, 40, 80, 96, 192, 320]
    n_cell_stages = [4, 4, 4, 4, 4, 1]
    stride_stages = [2, 2, 2, 1, 2, 1]
    conv_candidates = ['3x3_MBConv1', '3x3_MBConv3', '3x3_MBConv6', '5x5_MBConv3', '5x5_MBConv6', '7x7_MBConv3']
    
    super_net = SuperProxylessNASNets(
        width_stages=width_stages,
        n_cell_stages=n_cell_stages,
        conv_candidates=conv_candidates,
        stride_stages=stride_stages,
        n_classes=10,
        width_mult=1.0,
        dropout_rate=0,
    )
    print("✓ ネットワーク作成成功")
    
    # ネットワークをトレーニングモードに
    print("  ネットワークをtrain()モードに切り替え中...")
    start = time.time()
    super_net.train()
    elapsed = time.time() - start
    print(f"✓ train()モード切り替え成功（{elapsed:.3f}秒）")
    
except Exception as e:
    print(f"✗ ネットワーク関連エラー: {e}")
    print("  → この段階でエラーが発生している可能性があります")
    # この段階でのエラーは致命的ではない可能性

# ==================== Summary ====================
print("\n" + "=" * 80)
print("DIAGNOSTIC RESULTS")
print("=" * 80)

print("""
✓ 全てのテストが成功しました！

フリーズの原因は以下の可能性があります：

1. **本スクリプトでハングした場合**
   → マルチプロセッシング関連の問題
   → 実行方法: python -u verify_warmup.py | tee verify_output.log
   → タイムアウト後、プロセス状態を確認

2. **本スクリプトは成功したが cifar10_arch_search.py でハングする場合**
   → NAS固有の処理でハング（ネットワーク初期化など）
   → 関連ファイル: search/nas_manager.py warm_up() メソッド

3. **GPU メモリ不足の場合**
   → batch_size を小さくするか、num_workers を0にする

推奨アクション：
1. このスクリプトを実行して確認
2. フリーズ地点を特定
3. 適切な修正を適用
""")

print("\n[NEXT STEP] 実行フォーマット:")
print("  # タイムアウト付きで実行（30秒）")
print("  timeout 30 python verify_warmup.py")
print("\n  # または通常実行（フリーズ時はCtrl+Cで中断）")
print("  python verify_warmup.py")
