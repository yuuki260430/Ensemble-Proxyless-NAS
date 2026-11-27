# CIFAR-100 pilot 2 (Zero space, Exp6/pilot2_queue.sh)
# (1) do distinct near-front architectures all reach high accuracy after retraining?
# (2) do they give more diversity than seeds (disagreement, ensemble acc / NLL)?
# (3) does the supernet accuracy rank the retrained accuracy (16 archs incl. 4 lower-ranked ones)?
# (4) do the Hamming distance / pool similarity S predict the pairwise diversity (66 pairs of 12 top archs)?
# usage: cd search && uv run python cifar100_pilot2_analysis.py  (writes Exp6/pilot2_analysis.json; EXP=Exp7 for Exp7/)
import itertools
import json
import os

import numpy as np
import torch

from cifar10_ensemble_eval import load_net, run_setting, build_loader, predict

ROOT = os.path.abspath('..')
EXP = os.environ.get('EXP', 'Exp6')  # Exp6: ImageNet-style retraining recipe, Exp7: CIFAR-standard recipe
BASE = os.path.join(ROOT, EXP, 'evo_zero')
device = torch.device('cuda:0')


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def kendall(a, b):
    s = [np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in itertools.combinations(range(len(a)), 2)]
    return float(np.mean(s))


def group(name, n):
    info = json.load(open(os.path.join(BASE, name, 'archs.json')))
    return [(os.path.join(BASE, name, a['name']), tuple(a['arch'])) for a in info['archs']][:n]


even = group('even6_min76', 6)
near = group('near_front', 6)
rank = group('rank_check', 4)
seeds = [even[2][0]] + [os.path.join(BASE, 'seed_even02/s%d' % k) for k in (1, 2, 3)]
archs = even + near + rank
evaluated = {tuple(r['arch']): r for r in json.load(open(os.path.join(BASE, 'all_evaluated.json')))}
sup = np.array([evaluated[a]['supernet_acc'] for _, a in archs])
flops = np.array([evaluated[a]['flops_M'] for _, a in archs])

sims = {}
for tag in ['feature', 'prob']:
    info = json.load(open(os.path.join(BASE, 'dpp_pilot_all_%s' % tag, 'archs.json')))
    pool = [tuple(a) for a in info['pool_archs']]
    idx = [pool.index(a) for _, a in archs]
    sims[tag] = np.array(info['dpp']['pool_similarity'])[np.ix_(idx, idx)]

out = {'archs': [list(a) for _, a in archs], 'dirs': [d for d, _ in archs], 'supernet_acc': sup.tolist(),
       'flops_M': flops.tolist()}
for split in ['valid', 'test']:
    dirs = [d for d, _ in archs] + seeds[1:]
    dataset, vs = run_setting(dirs)
    loader = build_loader(split, 500, 4, dataset, vs)
    ds = loader.dataset
    labels = torch.tensor(ds.dataset.targets)[ds.indices] if hasattr(ds, 'indices') else torch.tensor(ds.targets)
    probs = {d: predict(load_net(d, 'last'), loader, device) for d in dirs}
    preds = {d: p.argmax(1) for d, p in probs.items()}
    err = {d: (preds[d] != labels).float().mean().item() for d in dirs}
    acc = {d: 100 - 100 * err[d] for d in dirs}

    def ndis(a, b):
        dis = (preds[a] != preds[b]).float().mean().item()
        return dis * 100, dis / (err[a] + err[b])

    def ens(ms):
        m = torch.stack([probs[d] for d in ms]).mean(0)
        return ((m.argmax(1) == labels).float().mean().item() * 100,
                torch.nn.functional.nll_loss(torch.log(m.clamp_min(1e-12)), labels).item())

    r = {}
    real = np.array([acc[d] for d, _ in archs])
    r['single'] = real.tolist()
    r['single_seed'] = [acc[d] for d in seeds]
    print('== %s' % split)
    for name, g in [('even6', even), ('near_front', near), ('rank_check', rank)]:
        print('  single %-10s %s' % (name, ' '.join('%.2f' % acc[d] for d, _ in g)))
    print('  single seeds      %s' % ' '.join('%.2f' % acc[d] for d in seeds))
    r['spearman_sup_vs_real_16'] = spearman(sup, real)
    r['kendall_sup_vs_real_16'] = kendall(sup, real)
    r['kendall_sup_vs_real_top12'] = kendall(sup[:12], real[:12])
    print('  supernet acc vs retrained acc: Spearman %.2f, Kendall %.2f (16 archs) | Kendall %.2f (12 top archs)' % (
        r['spearman_sup_vs_real_16'], r['kendall_sup_vs_real_16'], r['kendall_sup_vs_real_top12']))

    pairs = {'near_front': list(itertools.combinations([d for d, _ in near], 2)),
             'even6': list(itertools.combinations([d for d, _ in even], 2)),
             'seed': list(itertools.combinations(seeds, 2))}
    for name, ps in pairs.items():
        v = [ndis(a, b) for a, b in ps]
        r['dis_' + name] = [x[0] for x in v]
        r['ndis_' + name] = [x[1] for x in v]
        print('  pairs %-10s disagreement %.2f +- %.2f | normalized %.3f +- %.3f (%.3f-%.3f)' % (
            name, np.mean(r['dis_' + name]), np.std(r['dis_' + name]), np.mean(r['ndis_' + name]),
            np.std(r['ndis_' + name]), min(r['ndis_' + name]), max(r['ndis_' + name])))

    top = list(range(12))
    tp = list(itertools.combinations(top, 2))
    nd = [ndis(archs[i][0], archs[j][0])[1] for i, j in tp]
    ham = [sum(x != y for x, y in zip(archs[i][1], archs[j][1])) for i, j in tp]
    r['top12_ndis'] = nd
    r['top12_hamming'] = ham
    r['spearman_hamming_vs_ndis'] = spearman(ham, nd)
    for tag, S in sims.items():
        r['spearman_S_vs_ndis_' + tag] = spearman([S[i, j] for i, j in tp], nd)
    print('  66 pairs of the 12 top archs: Spearman(Hamming, ndis) %+.2f | (S feature, ndis) %+.2f | (S prob, ndis) %+.2f'
          % (r['spearman_hamming_vs_ndis'], r['spearman_S_vs_ndis_feature'], r['spearman_S_vs_ndis_prob']))

    for name, g in [('near_front', [d for d, _ in near]), ('even6', [d for d, _ in even]),
                    ('top12', [d for d, _ in archs[:12]])]:
        e = [ens(list(c)) for c in itertools.combinations(g, 4)]
        r['ens4_' + name] = e
        print('  K=4 %-10s acc %.2f (%.2f-%.2f) | NLL %.4f (%.4f-%.4f)  [%d subsets]' % (
            name, np.mean([x[0] for x in e]), min(x[0] for x in e), max(x[0] for x in e),
            np.mean([x[1] for x in e]), min(x[1] for x in e), max(x[1] for x in e), len(e)))
    r['ens4_seed'] = ens(seeds)
    print('  K=4 seed       acc %.2f | NLL %.4f' % r['ens4_seed'])
    r['ens6_near_front'] = ens([d for d, _ in near])
    r['ens6_even6'] = ens([d for d, _ in even])
    print('  K=6 near_front acc %.2f NLL %.4f | K=6 even6 acc %.2f NLL %.4f' % (r['ens6_near_front'] + r['ens6_even6']))
    out[split] = r
json.dump(out, open(os.path.join(ROOT, EXP, 'pilot2_analysis.json'), 'w'), indent=1)
print('Saved to %s/pilot2_analysis.json' % EXP)
