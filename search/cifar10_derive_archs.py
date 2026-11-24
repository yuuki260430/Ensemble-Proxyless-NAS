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

from data_providers.cifar10 import get_cifar_provider
from models.super_nets.super_proxyless import SuperProxylessNASNets
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge
from utils import count_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--search_path', type=str, default=None, help='output dir of cifar10_arch_search.py')
parser.add_argument('--candidates', type=str, default=None,
                    help='FairNAS baseline: candidates.json of cifar10_fairnas_evo.py, used instead of alpha '
                         '(the supernet settings are read from <supernet_path>/supernet.config)')
parser.add_argument('--candidate_set', type=str, default='final_population',
                    help='[candidates] which list of candidates.json forms the pool (final_population, pareto_front, '
                         'or a list added by hand, e.g. near_front)')
parser.add_argument('--run_config', type=str, default=None,
                    help='[candidates] run.config for retraining (default for --search_path: learned_net/run.config)')
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
parser.add_argument('--select', type=str, default='order', choices=['order', 'dpp', 'even'],
                    help='order: take the first n_archs valid candidates; '
                         'dpp: collect pool_size candidates and pick n_archs by NEAS-style diversity score; '
                         'even: [candidates] n_archs evenly spaced by MACs (FairNAS baseline within the same pool)')
parser.add_argument('--save_init', action='store_true',
                    help='[candidates] also write the inherited supernet weights as arch_XX/init (fine-tuning); '
                         'by default no init is written, so retraining starts from scratch')
parser.add_argument('--pool_size', type=int, default=30, help='[dpp] number of candidates to score')
parser.add_argument('--dpp_beta', type=float, default=None,
                    help='[dpp] similarity scale beta in exp(-beta * ||v_p - v_q||); default 1 / median distance')
parser.add_argument('--dpp_gamma', type=float, default=1.0,
                    help='[dpp] quality exponent (r ** gamma); larger favors accuracy over diversity')
parser.add_argument('--bn_batches', type=int, default=20, help='[dpp] train batches for BN recalibration')
parser.add_argument('--probe_size', type=int, default=2000, help='[dpp] train images for quality / features')
# options added 2026-11-19; the defaults reproduce the original selection exactly
parser.add_argument('--dpp_score', type=str, default='logdet', choices=['logdet', 'rank_sat'],
                    help='[dpp] logdet: greedy MAP of det(diag(r) S diag(r)) (original); rank_sat: DivBO-style '
                         'sum of ranks R_perf + w_k R_div with a saturating weight w_k = sat_beta (sigmoid(sat_tau k) - 0.5)')
parser.add_argument('--sat_beta', type=float, default=2.0, help='[rank_sat] maximum diversity weight is sat_beta / 2')
parser.add_argument('--sat_tau', type=float, default=1.0, help='[rank_sat] speed of saturation over greedy steps k')
parser.add_argument('--div_gain', type=str, default='logdet', choices=['logdet', 'min'],
                    help='[rank_sat] diversity of adding j: log det S over chosen + j (DPP), '
                         'or min_{i in chosen} (1 - S_ij) (DivBO Eq. 4)')
parser.add_argument('--dpp_quality', type=str, default='probe', choices=['probe', 'supernet_acc'],
                    help='[dpp] probe: accuracy on the probe images (original); '
                         'supernet_acc: [candidates] valid-split accuracy measured by the evolutionary search')
parser.add_argument('--dpp_diversity', type=str, default='feature', choices=['feature', 'prob'],
                    help='[dpp] distance used in S = E_n[exp(-beta d_n)]: feature: penultimate features (original); '
                         'prob: class probabilities (DivBO Eq. 3)')
parser.add_argument('--probe_split', type=str, default='train', choices=['train', 'valid'],
                    help='[dpp] images for probe accuracy / features: train subset (original) or the held-out valid split')
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
        conv_candidates=CONV_CANDIDATES, n_classes=get_cifar_provider(args.dataset).N_CLASSES, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, first_stride=args.first_stride,
        allow_zero=getattr(args, 'allow_zero', True),
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
    provider = get_cifar_provider(args.dataset)(train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                                                valid_size=valid_size, n_worker=args.n_worker)
    if args.probe_split == 'valid':
        # held-out split (never used to train the supernet weights); provider.valid.dataset shares its indexing
        source_indexes = sorted(provider.valid.sampler.indices)
    else:
        source_indexes = sorted(provider.train.sampler.indices)
    probe_indexes = np.random.RandomState(args.manual_seed).choice(
        source_indexes, min(args.probe_size, len(source_indexes)), replace=False)
    # provider.valid.dataset is the CIFAR train set with test-time transforms
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
    """ pooled penultimate features, predictions and class probabilities on the probe set """
    net.eval()
    features, preds, probs = [], [], []
    with torch.no_grad():
        for images, _ in loader:
            x = net.first_conv(images.to(device, non_blocking=True))
            for block in net.blocks:
                x = block(x)
            x = net.global_avg_pooling(net.feature_mix_layer(x)).flatten(1)
            logits = net.classifier(x)
            features.append(x)
            preds.append(logits.argmax(dim=1).cpu())
            probs.append(torch.softmax(logits, dim=1))
    return torch.cat(features), torch.cat(preds), torch.cat(probs)


def average_ranks(values, descending=True):
    """ 0-based ranks (0 = best), ties get the average rank """
    v = -np.asarray(values, dtype=np.float64) if descending else np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind='stable')
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v), dtype=np.float64)
    for value in np.unique(v):
        tied = v == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def saturating_weight(k, beta, tau):
    """ DivBO (Shen et al., NeurIPS 2022) Eq. 5, with the BO iteration t replaced by the greedy step k """
    return beta * (1.0 / (1.0 + np.exp(-tau * k)) - 0.5)


def log_det(matrix, indexes):
    sign, value = torch.linalg.slogdet(matrix[indexes][:, indexes])
    return value.item() if sign > 0 else -float('inf')


def greedy_select(quality, sim, n, score='logdet', gamma=1.0, sat_beta=2.0, sat_tau=1.0, div_gain='logdet',
                  first=None):
    """
    Greedy selection of n candidates from quality (list, higher = better) and similarity sim (P x P tensor).
    Shared by dpp_select and the offline subset analysis (cifar10_subset_analysis.py); see dpp_select for the
    definitions. Returns (chosen indexes, per-step info for rank_sat, the DPP kernel L).
    """
    sim = torch.as_tensor(sim, dtype=torch.float64)
    P = len(quality)
    r = torch.tensor(quality, dtype=torch.float64).clamp_min(1e-6) ** gamma
    kernel = r[:, None] * sim * r[None, :]
    chosen = [int(torch.argmax(r))] if first is None else [first]
    steps = []
    while len(chosen) < min(n, P):
        rest = [j for j in range(P) if j not in chosen]
        if score == 'logdet':
            gains = [log_det(kernel, chosen + [j]) for j in rest]
            chosen.append(rest[int(np.argmax(gains))])
            continue
        # rank_sat: diversity of each remaining candidate w.r.t. the chosen set, on S only (quality is ranked apart)
        if div_gain == 'logdet':
            div = [log_det(sim, chosen + [j]) for j in rest]
        else:
            div = [min(1.0 - sim[i, j].item() for i in chosen) for j in rest]
        r_perf = average_ranks([quality[j] for j in rest])
        r_div = average_ranks(div)
        w = saturating_weight(len(chosen), sat_beta, sat_tau)
        total = r_perf + w * r_div
        best = min(range(len(rest)), key=lambda i: (total[i], -quality[rest[i]]))  # ties -> higher quality
        steps.append({'k': len(chosen), 'w': w, 'chosen': rest[best], 'R_perf': r_perf[best], 'R_div': r_div[best]})
        chosen.append(rest[best])
    return chosen, steps, kernel


def dpp_select(pool, valid_size, args, force_first, ext_quality=None):
    """
    NEAS diversity score (Chen et al., CVPR 2021, Eq. 4, 5, 7, 8) applied to whole architectures:
        r_p    = ACC_train'(p) ** gamma                           (quality)
        S_pq   = E_n[ exp(-beta * ||v_p(n) - v_q(n)||_2) ]        (similarity of penultimate features)
        L      = diag(r) S diag(r),  score(subset) = det(L_subset)
    The subset of n_archs is chosen by greedy MAP inference.
    NOTE: NEAS scores per-layer operator combinations to shrink the search space; here the same
    kernel is used at the architecture level to reject redundant candidates before retraining.

    Options (defaults = the original method above):
      --dpp_quality supernet_acc : r_p from ext_quality (valid-split accuracy of the evolutionary search, in %)
      --dpp_diversity prob       : v_p(n) = class probabilities instead of penultimate features
      --dpp_score rank_sat       : instead of det(L), at greedy step k (k = number already chosen) add the j that
                                   minimizes R_perf(j) + w_k R_div(j), where R are 0-based ranks among the remaining
                                   candidates (0 = best) and w_k = sat_beta (sigmoid(sat_tau k) - 0.5) (DivBO Eq. 5).
                                   Ranks make the balance independent of the scale of accuracy and similarity.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bn_loader, probe_loader, probe_labels = build_dpp_data(valid_size, args)

    probe_acc, vectors = [], []
    for idx, (arch, net, flops) in enumerate(pool):
        net = copy.deepcopy(net).to(device)
        recalibrate_bn(net, bn_loader, args.bn_batches, device)
        feats, preds, probs = probe_features(net, probe_loader, device)
        probe_acc.append((preds == probe_labels).float().mean().item())
        vectors.append(probs if args.dpp_diversity == 'prob' else feats)
        print('\t[dpp] candidate %d/%d\tprobe acc %.2f\tFLOPs %.1fM' % (idx + 1, len(pool), probe_acc[-1] * 100, flops))
        del net

    if args.dpp_quality == 'supernet_acc':
        assert ext_quality is not None, '--dpp_quality supernet_acc needs --candidates'
        quality = [q / 100.0 for q in ext_quality]
    else:
        quality = probe_acc

    P = len(pool)
    # per-sample distances ||v_p(n) - v_q(n)||_2 for every pair
    pair_dist = {(p, q): (vectors[p] - vectors[q]).norm(dim=1) for p in range(P) for q in range(p + 1, P)}
    beta = args.dpp_beta
    if beta is None:
        beta = 1.0 / torch.stack([d.mean() for d in pair_dist.values()]).median().clamp_min(1e-12).item()
    sim = torch.eye(P, dtype=torch.float64)
    for (p, q), d in pair_dist.items():
        sim[p, q] = sim[q, p] = torch.exp(-beta * d).mean().item()
    chosen, steps, kernel = greedy_select(
        quality, sim, args.n_archs, score=args.dpp_score, gamma=args.dpp_gamma, sat_beta=args.sat_beta,
        sat_tau=args.sat_tau, div_gain=args.div_gain, first=0 if force_first else None)

    info = {
        'beta': beta, 'gamma': args.dpp_gamma, 'pool_size': P, 'score': args.dpp_score,
        'quality_source': args.dpp_quality, 'diversity': args.dpp_diversity, 'probe_split': args.probe_split,
        'pool_quality': quality, 'pool_probe_acc': probe_acc, 'pool_similarity': sim.tolist(),
        'chosen_pool_indexes': chosen, 'log_det': log_det(kernel, chosen), 'log_det_S': log_det(sim, chosen),
    }
    if args.dpp_score == 'rank_sat':
        info.update({'sat_beta': args.sat_beta, 'sat_tau': args.sat_tau, 'div_gain': args.div_gain, 'steps': steps})
        for s_ in steps:
            print('\t[rank_sat] k=%d w=%.3f -> %d (R_perf %.1f, R_div %.1f)' % (
                s_['k'], s_['w'], s_['chosen'], s_['R_perf'], s_['R_div']))
    print('[dpp] beta %.4g\tchosen pool indexes %s\tlog det %.3f' % (beta, chosen, info['log_det']))
    return [pool[i] + (probe_acc[i],) for i in chosen], info


def topk_archs(log_probs, k):
    """ exact top-k joint log-prob via beam search (edges are independent) """
    beams = [((), 0.0)]
    for lp in log_probs:
        beams = [(arch + (i,), score + lp[i]) for arch, score in beams for i in range(len(lp))]
        beams = sorted(beams, key=lambda b: -b[1])[:k]
    return [arch for arch, _ in beams]


def export(selected, out_path, run_config_path, info, save_init=True, dataset='cifar10'):
    """ write arch_XX/{net.config, run.config, init} and archs.json; selected = [(arch, net, flops, quality, extra)] """
    os.makedirs(out_path, exist_ok=True)
    summary = []
    for idx, (arch, net, flops, quality, extra) in enumerate(selected):
        arch_dir = os.path.join(out_path, 'arch_%02d' % idx)
        os.makedirs(arch_dir, exist_ok=True)
        json.dump(net.config, open(os.path.join(arch_dir, 'net.config'), 'w'), indent=4)
        shutil.copy(run_config_path, os.path.join(arch_dir, 'run.config'))
        if save_init:
            torch.save({'state_dict': net.state_dict(), 'dataset': dataset}, os.path.join(arch_dir, 'init'))
        summary.append(dict(name='arch_%02d' % idx, arch=list(arch), flops_M=round(flops, 2),
                            params_M=round(count_parameters(net) / 1e6, 3), **extra))
        if quality is not None:
            summary[-1]['probe_acc'] = quality
        print('%s\tFLOPs %.1fM\tparams %.2fM\t%s' % (summary[-1]['name'], flops, summary[-1]['params_M'], extra))
    archs = [s_[0] for s_ in selected]
    hamming = [[int(sum(x != y for x, y in zip(a, b))) for b in archs] for a in archs]
    json.dump(dict(info, archs=summary, pairwise_hamming=hamming), open(os.path.join(out_path, 'archs.json'), 'w'),
              indent=4)
    print('Exported %d architectures to %s' % (len(selected), out_path))


def main_candidates(args):
    """ FairNAS baseline: pool = population / Pareto front of the evolutionary search (no alpha) """
    cand = json.load(open(args.candidates))
    cfg = json.load(open(os.path.join(cand['supernet_path'], 'supernet.config')))
    for key in ['width_stages', 'n_cell_stages', 'stride_stages', 'first_stride', 'width_mult',
                'bn_momentum', 'bn_eps', 'allow_zero']:
        setattr(args, key, cfg[key])
    args.dataset = cfg.get('dataset', 'cifar10')  # supernets trained before CIFAR-100 support have no key
    assert json.load(open(args.run_config)).get('dataset', 'cifar10') == args.dataset, 'run_config dataset mismatch'
    if args.dropout is None:
        # same classifier dropout as the Pareto-front export of cifar10_fairnas_evo.py (supernet itself uses 0)
        evo_dropout = cand['args'].get('dropout')
        args.dropout = evo_dropout if evo_dropout is not None else cfg['dropout']
    assert list(cfg['conv_candidates']) == CONV_CANDIDATES
    assert args.run_config is not None, '--run_config is required with --candidates'
    ckpt_path = args.checkpoint or cand['checkpoint']

    super_net = build_super_net(args)
    model_dict = super_net.state_dict()
    model_dict.update(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    super_net.load_state_dict(model_dict)
    print('Loaded supernet from %s' % ckpt_path)

    pool = []
    for c in cand[args.candidate_set]:
        if (args.min_flops is not None and c['flops_M'] < args.min_flops) or \
                (args.max_flops is not None and c['flops_M'] > args.max_flops):
            continue
        arch = tuple(c['arch'])
        pool.append((arch, to_normal_net(super_net, arch), net_flops(to_normal_net(super_net, arch)) / 1e6,
                     {'supernet_acc': c['supernet_acc']}))
    print('%d candidates in the pool (%s)' % (len(pool), args.candidate_set))

    dpp_info = None
    if args.select == 'dpp':
        valid_size = json.load(open(args.run_config))['valid_size']
        chosen, dpp_info = dpp_select([p[:3] for p in pool], valid_size, args, force_first=False,
                                      ext_quality=[p[3]['supernet_acc'] for p in pool])
        extra = {p[0]: p[3] for p in pool}
        selected = [(a, n, f, q, extra[a]) for a, n, f, q in chosen]
    elif args.select == 'even':
        # same rule as the Pareto export of cifar10_fairnas_evo.py, restricted to the filtered pool
        by_flops = sorted(pool, key=lambda p: p[2])
        K = min(args.n_archs, len(by_flops))
        selected = [(a, n, f, None, e) for a, n, f, e in
                    [by_flops[int(round(i))] for i in np.linspace(0, len(by_flops) - 1, K)]]
    else:
        selected = [(a, n, f, None, e) for a, n, f, e in pool[:args.n_archs]]
    export(selected, args.out_path, args.run_config, {
        'candidates': args.candidates, 'candidate_set': args.candidate_set, 'checkpoint': ckpt_path,
        'select': args.select, 'min_flops': args.min_flops, 'max_flops': args.max_flops, 'dropout': args.dropout,
        'pool_size': len(pool), 'dpp': dpp_info,
        # pool order = order of pool_quality / pool_similarity (used by cifar10_subset_analysis.py)
        'pool_archs': [list(p[0]) for p in pool], 'pool_supernet_acc': [p[3]['supernet_acc'] for p in pool],
        'pool_flops_M': [round(p[2], 2) for p in pool],
    }, save_init=args.save_init, dataset=args.dataset)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.manual_seed)
    # BN recalibration draws its batches from a shuffled train loader; without this the probe accuracy (and hence the
    # DPP selection) changed from run to run (fixed 2026-11-19, after the Exp5 dpp5_min76 selection)
    torch.manual_seed(args.manual_seed)
    if args.candidates is not None:
        main_candidates(args)
        raise SystemExit
    assert args.search_path is not None, '--search_path or --candidates is required'

    learned_net_path = os.path.join(args.search_path, 'learned_net')
    args.dataset = json.load(open(os.path.join(learned_net_path, 'run.config'))).get('dataset', 'cifar10')
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
        torch.save({'state_dict': net.state_dict(), 'dataset': args.dataset}, os.path.join(arch_dir, 'init'))
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
