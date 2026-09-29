# Derive multiple architectures from a single ProxylessNAS search (CIFAR-10)
# Each derived architecture is exported in the same format as `learned_net`
# (net.config / run.config / init), so it can be retrained with cifar10_run_exp.py.

import argparse
import copy
import itertools
import json
import os
import shutil

import numpy as np
import torch

from data_providers.cifar10 import CIFAR10DataProvider
from models.super_nets.super_proxyless import SuperProxylessNASNets
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge
from utils import count_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--search_path', type=str, required=True, help='output dir of cifar10_arch_search.py')
parser.add_argument('--checkpoint', type=str, default=None, help='default: <search_path>/checkpoint/checkpoint.pth.tar')
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--n_archs', type=int, default=5)
parser.add_argument('--mode', type=str, default='topk', choices=['topk', 'sample'],
                    help='topk: K most probable architectures; sample: sample from softmax(alpha / temperature)')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--no_argmax', action='store_true', help='do not force the argmax architecture as arch_00')
parser.add_argument('--min_flops', type=float, default=None, help='MFLOPs')
parser.add_argument('--max_flops', type=float, default=None, help='MFLOPs')
parser.add_argument('--min_hamming', type=int, default=0,
                    help='minimum number of differing edges between any two selected archs (diversity)')
parser.add_argument('--max_tries', type=int, default=10000)
parser.add_argument('--manual_seed', type=int, default=0)

""" selection among candidates """
parser.add_argument('--select', type=str, default='order', choices=['order', 'dpp'],
                    help='order: take the first n_archs valid candidates; '
                         'dpp: collect pool_size candidates and pick n_archs by NEAS-style diversity score')
parser.add_argument('--pool_size', type=int, default=30, help='[dpp] number of candidates to score')
parser.add_argument('--dpp_beta', type=float, default=None,
                    help='[dpp] similarity scale beta in exp(-beta * ||v_p - v_q||); default 1 / median distance')
parser.add_argument('--dpp_gamma', type=float, default=1.0,
                    help='[dpp] quality exponent (r ** gamma); larger favors accuracy over diversity')
parser.add_argument('--bn_batches', type=int, default=20, help='[dpp] train batches for BN recalibration')
parser.add_argument('--probe_size', type=int, default=2000, help='[dpp] train images for quality / features')
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--gpu', type=str, default='0')

""" supernet config (must match the search) """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='1,2,2,1,2,1')
parser.add_argument('--first_stride', type=int, default=1)
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=None, help='default: taken from <search_path>/learned_net/net.config')

CONV_CANDIDATES = [
    '3x3_MBConv3', '3x3_MBConv6',
    '5x5_MBConv3', '5x5_MBConv6',
    '7x7_MBConv3', '7x7_MBConv6',
]


def build_super_net(args):
    return SuperProxylessNASNets(
        width_stages=[int(v) for v in args.width_stages.split(',')],
        n_cell_stages=[int(v) for v in args.n_cell_stages.split(',')],
        stride_stages=[int(v) for v in args.stride_stages.split(',')],
        conv_candidates=CONV_CANDIDATES, n_classes=10, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, first_stride=args.first_stride,
    )


def to_normal_net(super_net, arch):
    """ replace each MixedEdge with the candidate op given by `arch` (one index per MixedEdge) """
    net = copy.deepcopy(super_net)
    edges = iter(arch)
    for block in net.blocks:
        if isinstance(block.mobile_inverted_conv, MixedEdge):
            block.mobile_inverted_conv = block.mobile_inverted_conv.candidate_ops[next(edges)]
    return ProxylessNASNets(net.first_conv, list(net.blocks), net.feature_mix_layer, net.classifier)


def net_flops(net):
    net.eval()
    with torch.no_grad():
        flops, _ = net.get_flops(torch.zeros(1, 3, 32, 32))
    return flops


def build_dpp_data(valid_size, args):
    """ BN recalibration loader and a fixed probe set, both from the training split (not the valid split) """
    provider = CIFAR10DataProvider(train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                   valid_size=valid_size, n_worker=args.n_worker)
    train_indexes = sorted(provider.train.sampler.indices)
    probe_indexes = np.random.RandomState(args.manual_seed).choice(train_indexes, args.probe_size, replace=False)
    # provider.valid.dataset is the CIFAR-10 train set with test-time transforms
    probe_set = torch.utils.data.Subset(provider.valid.dataset, sorted(probe_indexes.tolist()))
    probe_loader = torch.utils.data.DataLoader(probe_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.n_worker, pin_memory=True)
    probe_labels = torch.tensor(provider.valid.dataset.targets)[probe_set.indices]
    return provider.train, probe_loader, probe_labels


def recalibrate_bn(net, loader, n_batches, device):
    """ supernet BN statistics are shared by all paths, so recompute them for each subnet """
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average
    net.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            net(images.to(device, non_blocking=True))


def probe_features(net, loader, device):
    """ pooled penultimate features and predictions on the probe set """
    net.eval()
    features, preds = [], []
    with torch.no_grad():
        for images, _ in loader:
            x = net.first_conv(images.to(device, non_blocking=True))
            for block in net.blocks:
                x = block(x)
            x = net.global_avg_pooling(net.feature_mix_layer(x)).flatten(1)
            features.append(x)
            preds.append(net.classifier(x).argmax(dim=1).cpu())
    return torch.cat(features), torch.cat(preds)


def dpp_select(pool, valid_size, args, force_first):
    """
    NEAS diversity score (Chen et al., CVPR 2021, Eq. 4, 5, 7, 8) applied to whole architectures:
        r_p    = ACC_train'(p) ** gamma                           (quality)
        S_pq   = E_n[ exp(-beta * ||v_p(n) - v_q(n)||_2) ]        (similarity of penultimate features)
        L      = diag(r) S diag(r),  score(subset) = det(L_subset)
    The subset of n_archs is chosen by greedy MAP inference.
    NOTE: NEAS scores per-layer operator combinations to shrink the search space; here the same
    kernel is used at the architecture level to reject redundant candidates before retraining.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bn_loader, probe_loader, probe_labels = build_dpp_data(valid_size, args)

    quality, features = [], []
    for idx, (arch, net, flops) in enumerate(pool):
        net = copy.deepcopy(net).to(device)
        recalibrate_bn(net, bn_loader, args.bn_batches, device)
        feats, preds = probe_features(net, probe_loader, device)
        quality.append((preds == probe_labels).float().mean().item())
        features.append(feats)
        print('\t[dpp] candidate %d/%d\tprobe acc %.2f\tFLOPs %.1fM' % (idx + 1, len(pool), quality[-1] * 100, flops))
        del net

    P = len(pool)
    # per-sample feature distances ||v_p(n) - v_q(n)||_2 for every pair
    pair_dist = {(p, q): (features[p] - features[q]).norm(dim=1) for p in range(P) for q in range(p + 1, P)}
    beta = args.dpp_beta
    if beta is None:
        beta = 1.0 / torch.stack([d.mean() for d in pair_dist.values()]).median().clamp_min(1e-12).item()
    sim = torch.eye(P, dtype=torch.float64)
    for (p, q), d in pair_dist.items():
        sim[p, q] = sim[q, p] = torch.exp(-beta * d).mean().item()
    r = torch.tensor(quality, dtype=torch.float64).clamp_min(1e-6) ** args.dpp_gamma
    kernel = r[:, None] * sim * r[None, :]

    def log_det(indexes):
        sign, value = torch.linalg.slogdet(kernel[indexes][:, indexes])
        return value.item() if sign > 0 else -float('inf')

    chosen = [0] if force_first else [int(torch.argmax(r))]
    while len(chosen) < min(args.n_archs, P):
        rest = [j for j in range(P) if j not in chosen]
        gains = [log_det(chosen + [j]) for j in rest]
        chosen.append(rest[int(np.argmax(gains))])

    info = {
        'beta': beta, 'gamma': args.dpp_gamma, 'pool_size': P,
        'pool_quality': quality, 'pool_similarity': sim.tolist(),
        'chosen_pool_indexes': chosen, 'log_det': log_det(chosen),
    }
    print('[dpp] beta %.4g\tchosen pool indexes %s\tlog det %.3f' % (beta, chosen, info['log_det']))
    return [pool[i] + (quality[i],) for i in chosen], info


def topk_archs(log_probs, k):
    """ exact top-k joint log-prob via beam search (edges are independent) """
    beams = [((), 0.0)]
    for lp in log_probs:
        beams = [(arch + (i,), score + lp[i]) for arch, score in beams for i in range(len(lp))]
        beams = sorted(beams, key=lambda b: -b[1])[:k]
    return [arch for arch, _ in beams]


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.manual_seed)

    learned_net_path = os.path.join(args.search_path, 'learned_net')
    if args.dropout is None:
        learned_config = json.load(open(os.path.join(learned_net_path, 'net.config')))
        args.dropout = learned_config['classifier']['dropout_rate']
    ckpt_path = args.checkpoint or os.path.join(args.search_path, 'checkpoint', 'checkpoint.pth.tar')

    super_net = build_super_net(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    super_net.load_state_dict(checkpoint['state_dict'])
    print('Loaded supernet from %s (epoch %s)' % (ckpt_path, checkpoint.get('epoch')))

    edges = super_net.redundant_modules
    op_names = [[op.module_str for op in m.candidate_ops] for m in edges]
    probs = [m.probs_over_ops.data.numpy().astype(np.float64) for m in edges]
    log_probs = [np.log(p) for p in probs]

    def arch_log_prob(arch):
        return float(sum(lp[i] for lp, i in zip(log_probs, arch)))

    # number of valid candidates to collect before selection
    target = args.pool_size if args.select == 'dpp' else args.n_archs

    # candidate generator
    if args.mode == 'topk':
        # over-generate so that FLOPs / diversity filtering still leaves enough archs
        constrained = args.min_flops is not None or args.max_flops is not None or args.min_hamming > 0
        candidates = iter(topk_archs(log_probs, args.max_tries if constrained else target))
    else:
        def sampler():
            for _ in range(args.max_tries):
                arch = []
                for lp in log_probs:
                    p = np.exp(lp / args.temperature)
                    arch.append(int(np.random.choice(len(p), p=p / p.sum())))
                yield tuple(arch)
        candidates = sampler()

    argmax_arch = tuple(int(np.argmax(p)) for p in probs)
    if not args.no_argmax:
        candidates = itertools.chain([argmax_arch], candidates)

    selected, seen = [], set()
    for arch in candidates:
        if len(selected) >= target:
            break
        if arch in seen:
            continue
        seen.add(arch)
        if any(sum(x != y for x, y in zip(arch, a)) < args.min_hamming for a, _, _ in selected):
            continue
        net = to_normal_net(super_net, arch)
        flops = net_flops(net) / 1e6
        if (args.min_flops is not None and flops < args.min_flops) or \
                (args.max_flops is not None and flops > args.max_flops):
            continue
        selected.append((arch, net, flops))
    if len(selected) < target:
        print('Warning: only %d architectures satisfy the constraints' % len(selected))

    dpp_info = None
    if args.select == 'dpp':
        run_config = json.load(open(os.path.join(learned_net_path, 'run.config')))
        force_first = not args.no_argmax and len(selected) > 0 and selected[0][0] == argmax_arch
        selected, dpp_info = dpp_select(selected, run_config['valid_size'], args, force_first)
    else:
        selected = [s_ + (None,) for s_ in selected]

    # export
    os.makedirs(args.out_path, exist_ok=True)
    summary = []
    for idx, (arch, net, flops, quality) in enumerate(selected):
        arch_dir = os.path.join(args.out_path, 'arch_%02d' % idx)
        os.makedirs(arch_dir, exist_ok=True)
        json.dump(net.config, open(os.path.join(arch_dir, 'net.config'), 'w'), indent=4)
        shutil.copy(os.path.join(learned_net_path, 'run.config'), os.path.join(arch_dir, 'run.config'))
        torch.save({'state_dict': net.state_dict(), 'dataset': 'cifar10'}, os.path.join(arch_dir, 'init'))
        summary.append({
            'name': 'arch_%02d' % idx,
            'ops': [op_names[e][i] for e, i in enumerate(arch)],
            'log_prob': arch_log_prob(arch),
            'flops_M': round(flops, 2),
            'params_M': round(count_parameters(net) / 1e6, 3),
            'hamming_to_argmax': int(sum(a != b for a, b in zip(arch, argmax_arch))),
        })
        if quality is not None:
            summary[-1]['probe_acc'] = quality
        print('%s\tlog_prob %.3f\tFLOPs %.1fM\tparams %.2fM\tdiff from argmax: %d edges' % (
            summary[-1]['name'], summary[-1]['log_prob'], flops, summary[-1]['params_M'],
            summary[-1]['hamming_to_argmax']))

    archs = [a[0] for a in selected]
    hamming = [[int(sum(x != y for x, y in zip(a, b))) for b in archs] for a in archs]
    entropy = float(sum(-(p * np.log(p)).sum() for p in probs))
    json.dump({
        'search_path': args.search_path, 'checkpoint': ckpt_path, 'mode': args.mode,
        'temperature': args.temperature, 'n_edges': len(edges), 'alpha_entropy': entropy,
        'select': args.select, 'archs': summary, 'pairwise_hamming': hamming, 'dpp': dpp_info,
    }, open(os.path.join(args.out_path, 'archs.json'), 'w'), indent=4)
    print('alpha entropy (sum over %d edges): %.3f' % (len(edges), entropy))
    print('Exported %d architectures to %s' % (len(selected), args.out_path))
