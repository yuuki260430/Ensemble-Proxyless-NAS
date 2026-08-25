# ProxylessNAS: タスクとハードウェア上での直接的なニューラルアーキテクチャ探索（ProxylessNAS） [[arXiv]](https://arxiv.org/abs/1812.00332) [[ポスター]](../assets/ProxylessNAS_iclr_poster_final.pdf)
```bash
@inproceedings{
  cai2018proxylessnas,
  title={Proxyless{NAS}: Direct Neural Architecture Search on Target Task and Hardware},
  author={Han Cai and Ligeng Zhu and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://arxiv.org/pdf/1812.00332.pdf},
}
```

## ニュース
- 次世代の ProxylessNAS: [Once-for-All](https://github.com/mit-han-lab/once-for-all)（第3回および第4回 [Low-Power Computer Vision Challenge](https://lpcv.ai/competitions/2019) で1位）。
- Visual Wake Words Challenge（TF-lite トラック）で1位（@CVPR 2019）
- Low Power Image Recognition Challenge（LPIRC）分類トラックで3位（@CVPR 2019）

## 性能

プロキシを用いず、ターゲットの**タスク**および**ハードウェア**上で直接かつ効率的にニューラルネットワークのアーキテクチャ探索を行います！

![](../assets/proxyless_nas.png)

<p align="center">
    <img src="../assets/proxyless_bar.png" width="80%" />
</p>

![](../assets/proxyless_compare.png)

<table>
<tr>
    <th> モバイル設定 </th><th> GPU設定 </th>
</tr>
<tr>
    <td>
    <img src="../assets/proxyless_vs_mobilenet.png" width="100%" />
    </td>
<td>

| モデル                | Top-1    | Top-5    | レイテンシ | 
|----------------------|----------|----------|---------| 
| MobilenetV2          | 72.0     | 91.0     | 6.1ms   |
| ShufflenetV2(1.5)    | 72.6     | -        | 7.3ms   |
| ResNet-34            | 73.3     | 91.4     | 8.0ms   |
| MNasNet(当方実装)    | 74.0     | 91.8     | 6.1ms   | 
| ProxylessNAS (GPU)   | 75.1     | 92.5     | 5.1ms   |

</td>
</tr>
<tr>
    <th> ProxylessNAS（モバイル）はさまざまなレイテンシ設定下で一貫して MobileNetV2 を上回ります。  </th>
    <th> ProxylessNAS（GPU）は MobilenetV2 より3.1%高精度かつ20%高速です。 </th>
</tr> 

</td></tr> </table>


## 特化（Specialization）

従来は一つのモデルをすべてのプラットフォームにデプロイすることが一般的でしたが、それは最適とはいえません。効率を最大限に引き出すためには、各プラットフォーム向けにアーキテクチャを特化させるべきです。

![](../assets/specialization.jpg)
[![ビデオを見る](../assets/specialized_archs.png)](../assets/visualization.mp4)

## 要件
- Pytorch 1.0
- Python 3.6+

```bash
# モデルを訓練: <path>/run.config; <path>/net.config
python python imagenet_run_exp.py --path <path> --train

# Proxyless を訓練 (GPU)
python imagenet_run_exp.py --path Exp/proxyless_gpu --train --net proxyless_gpu --dropout 0.3

# Proxyless を訓練 (CPU)
python imagenet_run_exp.py --path Exp/proxyless_cpu --train --net proxyless_cpu --dropout 0.2

# Proxyless を訓練 (Mobile)
python imagenet_run_exp.py --path Exp/proxyless_mobile --train --net proxyless_mobile --dropout 0.1

# Proxyless を訓練 (Mobile-14)
python imagenet_run_exp.py --path Exp/proxyless_mobile_14 --train --net proxyless_mobile_14 --dropout 0.3
```

```bash
# Proxyless を評価 (GPU)
python imagenet_run_exp.py --path Exp/proxyless_gpu --net proxyless_gpu

# Proxyless を評価 (CPU)
python imagenet_run_exp.py --path Exp/proxyless_cpu --net proxyless_cpu

# Proxyless を評価 (Mobile)
python imagenet_run_exp.py --path Exp/proxyless_mobile --net proxyless_mobile

# Proxyless を評価 (Mobile-14)
python imagenet_run_exp.py --path Exp/proxyless_mobile_14 --net proxyless_mobile_14
```

```bash
# アーキテクチャ探索
please refer to imagenet_arch_search.py
```
