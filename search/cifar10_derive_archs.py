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


def topk_archs(log_probs, k):
    """ exact top-k joint log-prob via beam search (edges are independent) """
    beams = [((), 0.0)]
    for lp in log_probs:
        beams = [(arch + (i,), score + lp[i]) for arch, score in beams for i in range(len(lp))]
        beams = sorted(beams, key=lambda b: -b[1])[:k]
    return [arch for arch, _ in beams]


if __name__ == '__main__':
    args = parser.parse_args()
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

    # candidate generator
    if args.mode == 'topk':
        # over-generate so that FLOPs / diversity filtering still leaves enough archs
        constrained = args.min_flops is not None or args.max_flops is not None or args.min_hamming > 0
        candidates = iter(topk_archs(log_probs, args.max_tries if constrained else args.n_archs))
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
        if len(selected) >= args.n_archs:
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
    if len(selected) < args.n_archs:
        print('Warning: only %d architectures satisfy the constraints' % len(selected))

    # export
    os.makedirs(args.out_path, exist_ok=True)
    summary = []
    for idx, (arch, net, flops) in enumerate(selected):
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
        print('%s\tlog_prob %.3f\tFLOPs %.1fM\tparams %.2fM\tdiff from argmax: %d edges' % (
            summary[-1]['name'], summary[-1]['log_prob'], flops, summary[-1]['params_M'],
            summary[-1]['hamming_to_argmax']))

    archs = [a for a, _, _ in selected]
    hamming = [[int(sum(x != y for x, y in zip(a, b))) for b in archs] for a in archs]
    entropy = float(sum(-(p * np.log(p)).sum() for p in probs))
    json.dump({
        'search_path': args.search_path, 'checkpoint': ckpt_path, 'mode': args.mode,
        'temperature': args.temperature, 'n_edges': len(edges), 'alpha_entropy': entropy,
        'archs': summary, 'pairwise_hamming': hamming,
    }, open(os.path.join(args.out_path, 'archs.json'), 'w'), indent=4)
    print('alpha entropy (sum over %d edges): %.3f' % (len(edges), entropy))
    print('Exported %d architectures to %s' % (len(selected), args.out_path))
