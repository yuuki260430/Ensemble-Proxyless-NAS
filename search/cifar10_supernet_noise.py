# Measurement noise of the supernet accuracy used by NSGA-II (cifar10_fairnas_evo.py):
# re-evaluate architectures with different BN-recalibration image sets (--seeds) and report the per-arch spread
# and the rank agreement between seeds. Used to set the tolerance of the "near-front" candidate pool.
import argparse
import itertools
import json
import os

import numpy as np
import torch

from cifar10_fairnas_evo import build_super_net, SupernetEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('--evo_path', type=str, required=True, help='output dir of cifar10_fairnas_evo.py')
parser.add_argument('--n_top', type=int, default=60, help='best archs (supernet acc) among those >= --min_flops')
parser.add_argument('--n_random', type=int, default=60, help='random archs among those >= --min_flops')
parser.add_argument('--min_flops', type=float, default=76)
parser.add_argument('--seeds', type=str, default='0,1,2,3', help='0 reproduces the evolutionary search')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--valid_size', type=int, default=5000)
parser.add_argument('--bn_images', type=int, default=5000)


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0')
    cand = json.load(open(os.path.join(args.evo_path, 'candidates.json')))
    cfg = json.load(open(os.path.join(cand['supernet_path'], 'supernet.config')))
    args.dataset = cfg.get('dataset', 'cifar10')
    super_net = build_super_net(cfg)
    sd = super_net.state_dict()
    sd.update(torch.load(cand['checkpoint'], map_location='cpu')['state_dict'])
    super_net.load_state_dict(sd)

    rows = [r for r in json.load(open(os.path.join(args.evo_path, 'all_evaluated.json'))) if r['flops_M'] >= args.min_flops]
    rows.sort(key=lambda r: -r['supernet_acc'])
    rng = np.random.RandomState(0)
    rest = rows[args.n_top:]
    pick = rows[:args.n_top] + [rest[i] for i in rng.choice(len(rest), min(args.n_random, len(rest)), replace=False)]
    archs = [tuple(r['arch']) for r in pick]

    seeds = [int(s) for s in args.seeds.split(',')]
    acc = np.zeros((len(seeds), len(archs)))
    for k, s in enumerate(seeds):
        args.manual_seed = s
        ev = SupernetEvaluator(super_net, args, device)
        acc[k] = [ev.accuracy(a) for a in archs]
        print('seed %d done' % s, flush=True)
    stored = np.array([r['supernet_acc'] for r in pick])
    print('seed 0 reproduces the search: max |diff| %.3f' % np.abs(acc[seeds.index(0)] - stored).max() if 0 in seeds else '')
    std = acc.std(0, ddof=1)
    top = slice(0, args.n_top)
    print('per-arch std over %d BN sets: all %.3f (median %.3f, max %.3f) | top %d: %.3f' % (
        len(seeds), std.mean(), np.median(std), std.max(), args.n_top, std[top].mean()))
    diffs = [acc[i] - acc[j] for i, j in itertools.combinations(range(len(seeds)), 2)]
    print('|acc(seed i) - acc(seed j)| for the same arch: mean %.3f, 95%% %.3f' % (
        np.mean(np.abs(diffs)), np.percentile(np.abs(diffs), 95)))
    print('Spearman between seeds: all %s | top %d %s' % (
        ' '.join('%.2f' % spearman(acc[i], acc[j]) for i, j in itertools.combinations(range(len(seeds)), 2)),
        args.n_top,
        ' '.join('%.2f' % spearman(acc[i][top], acc[j][top]) for i, j in itertools.combinations(range(len(seeds)), 2))))
    json.dump({'archs': [list(a) for a in archs], 'flops_M': [r['flops_M'] for r in pick], 'seeds': seeds,
               'acc': acc.tolist(), 'stored': stored.tolist()},
              open(os.path.join(args.evo_path, 'supernet_noise.json'), 'w'))
