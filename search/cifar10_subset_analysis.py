# Offline analysis of the ensemble-member selection rule on already retrained models.
#
# Given n retrained architectures that all belong to one DPP candidate pool (cifar10_derive_archs.py --select dpp),
# enumerate every K-subset, compute its real ensemble accuracy / NLL (softmax average), and check how well each
# selection rule (original log det with gamma, DivBO-style rank_sat with (sat_beta, sat_tau)) would have chosen
# among these n models. Used to choose the balance between quality and diversity on a pilot run; the chosen
# setting is then fixed for the main experiments. Use the valid split for this, never the test split.

import argparse
import itertools
import json
import os

import numpy as np
import torch

from cifar10_derive_archs import greedy_select, log_det
from cifar10_ensemble_eval import load_net, run_setting, build_loader, predict

parser = argparse.ArgumentParser()
parser.add_argument('--pool_json', type=str, required=True,
                    help='archs.json of cifar10_derive_archs.py --select dpp (pool quality / similarity)')
parser.add_argument('--candidates', type=str, default=None,
                    help='candidates.json, only for archs.json written before pool_archs was stored')
parser.add_argument('--min_flops', type=float, default=None, help='with --candidates: the same filter as the DPP run')
parser.add_argument('--paths', type=str, nargs='+', required=True, help='retrained run dirs (arch_XX of derive_archs)')
parser.add_argument('--quality', type=str, default='pool', choices=['pool', 'supernet_acc'],
                    help='pool: quality stored by the DPP run; supernet_acc: valid accuracy of the evolutionary search')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--split', type=str, default='valid', choices=['valid', 'test'])
parser.add_argument('--gammas', type=str, default='1,4,10,40,100')
parser.add_argument('--sat_betas', type=str, default='0.5,1,2,4,8')
parser.add_argument('--sat_taus', type=str, default='0.25,0.5,1,2')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--output', type=str, default=None)


def rank_corr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pool_archs(info, args):
    if 'pool_archs' in info:
        return [tuple(a) for a in info['pool_archs']], info.get('pool_supernet_acc')
    assert args.candidates is not None, 'this archs.json has no pool_archs: pass --candidates (and --min_flops)'
    cand = json.load(open(args.candidates))
    rows = [c for c in cand[info.get('candidate_set', 'pareto_front')]
            if args.min_flops is None or c['flops_M'] >= args.min_flops]
    return [tuple(c['arch']) for c in rows], [c['supernet_acc'] for c in rows]


def resolve(path):
    """ follow SAME_AS (duplicate architecture trained once) and read the arch from the parent archs.json """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    same = os.path.join(path, 'SAME_AS')
    arch_src = path
    if os.path.exists(same):
        path = os.path.join(root, open(same).read().strip())
    summary = json.load(open(os.path.join(os.path.dirname(os.path.normpath(arch_src)), 'archs.json')))
    name = os.path.basename(os.path.normpath(arch_src))
    arch = [a['arch'] for a in summary['archs'] if a['name'] == name][0]
    return path, tuple(arch)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    info = json.load(open(args.pool_json))
    dpp = info['dpp']
    archs, supernet_acc = pool_archs(info, args)
    S_pool = np.array(dpp['pool_similarity'])
    q_pool = [a / 100.0 for a in supernet_acc] if args.quality == 'supernet_acc' else dpp['pool_quality']

    # unique retrained architectures and their pool indexes
    members = {}
    for p in args.paths:
        run_dir, arch = resolve(p)
        assert arch in archs, '%s is not in the pool' % p
        members.setdefault(arch, run_dir)
    idx = [archs.index(a) for a in members]
    dirs = list(members.values())
    n = len(dirs)
    print('%d unique architectures (pool indexes %s)' % (n, idx))

    dataset, valid_size = run_setting(dirs)
    loader = build_loader(args.split, args.batch_size, args.n_worker, dataset, valid_size)
    ds = loader.dataset
    labels = torch.tensor(ds.dataset.targets)[ds.indices] if hasattr(ds, 'indices') else torch.tensor(ds.targets)
    probs = [predict(load_net(d, 'last'), loader, device) for d in dirs]
    single = [(p.argmax(1) == labels).float().mean().item() * 100 for p in probs]

    # every K-subset of the retrained models
    subsets = list(itertools.combinations(range(n), args.k))
    acc, nll = [], []
    for c in subsets:
        mean = torch.stack([probs[i] for i in c]).mean(0)
        acc.append((mean.argmax(1) == labels).float().mean().item() * 100)
        nll.append(torch.nn.functional.nll_loss(torch.log(mean.clamp_min(1e-12)), labels).item())
    acc = np.array(acc)
    subset_index = {c: i for i, c in enumerate(subsets)}
    print('%d subsets of %d: ensemble acc %.2f-%.2f (mean %.2f, std %.2f)' % (
        len(subsets), args.k, acc.min(), acc.max(), acc.mean(), acc.std()))

    S = torch.tensor(S_pool[np.ix_(idx, idx)], dtype=torch.float64)
    q = [q_pool[i] for i in idx]

    def evaluate(chosen):
        c = tuple(sorted(chosen))
        a = acc[subset_index[c]]
        return {'chosen_local': list(c), 'chosen_pool': [idx[i] for i in c], 'acc': a,
                'nll': nll[subset_index[c]], 'percentile': float((acc < a).mean() * 100 + (acc == a).mean() * 50)}

    # (1) how well does each subset score rank the real ensemble accuracy?
    scores = {'sum quality': [sum(q[i] for i in c) for c in subsets],
              'log det S': [log_det(S, list(c)) for c in subsets]}
    for g in [float(v) for v in args.gammas.split(',')]:
        r = torch.tensor(q, dtype=torch.float64) ** g
        L = r[:, None] * S * r[None, :]
        scores['log det L (gamma %g)' % g] = [log_det(L, list(c)) for c in subsets]
    corr = {name: rank_corr(v, acc) for name, v in scores.items()}

    # (2) what would each greedy rule have picked among these n models?
    picks = {'top-K quality': evaluate(sorted(range(n), key=lambda i: -q[i])[:args.k])}
    for g in [float(v) for v in args.gammas.split(',')]:
        picks['logdet gamma=%g' % g] = evaluate(greedy_select(q, S, args.k, score='logdet', gamma=g)[0])
    for div_gain in ['logdet', 'min']:
        for b in [float(v) for v in args.sat_betas.split(',')]:
            for t in [float(v) for v in args.sat_taus.split(',')]:
                chosen = greedy_select(q, S, args.k, score='rank_sat', sat_beta=b, sat_tau=t, div_gain=div_gain)[0]
                picks['rank_sat %s beta=%g tau=%g' % (div_gain, b, t)] = evaluate(chosen)

    print('\nSpearman correlation between subset score and real ensemble acc (%s split)' % args.split)
    for name, v in corr.items():
        print('  %-24s %+.2f' % (name, v))
    print('\nGreedy picks (acc of the chosen subset; percentile among all %d subsets)' % len(subsets))
    print('  %-32s %.2f' % ('best possible', acc.max()))
    print('  %-32s %.2f' % ('mean of all subsets', acc.mean()))
    for name, v in picks.items():
        print('  %-32s %.2f  (%3.0f%%)  pool %s' % (name, v['acc'], v['percentile'], v['chosen_pool']))

    if args.output:
        json.dump({'args': vars(args), 'pool_indexes': idx, 'dirs': dirs, 'single_acc': single,
                   'subsets': [list(c) for c in subsets], 'subset_acc': acc.tolist(), 'subset_nll': nll,
                   'score_rank_corr': corr, 'picks': picks}, open(args.output, 'w'), indent=2)
        print('Saved to %s' % args.output)
