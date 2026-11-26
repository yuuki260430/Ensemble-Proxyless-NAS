# CIFAR-100, Zero space: every retrained architecture of Exp6/evo_zero (all MACs, SAME_AS resolved)
# (1) supernet accuracy / MACs vs retrained accuracy over the whole Pareto range
# (2) pairwise diversity vs MACs difference / Hamming, against seed pairs of a small (32.9M) and a large (84.6M) arch
# (3) ensembles under a MACs budget: distinct small archs vs seeds of one small arch vs one large model
# usage: cd search && uv run python cifar100_zero_all_analysis.py  (writes Exp6/zero_all_analysis.json)
import glob
import itertools
import json
import os

import numpy as np
import torch

from cifar10_ensemble_eval import load_net, run_setting, build_loader, predict

ROOT = os.path.abspath('..')
BASE = os.path.join(ROOT, 'Exp6/evo_zero')
device = torch.device('cuda:0')


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def kendall(a, b):
    return float(np.mean([np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in itertools.combinations(range(len(a)), 2)]))


evaluated = {tuple(r['arch']): r for r in json.load(open(os.path.join(BASE, 'all_evaluated.json')))}
arch_of = {}
for f in glob.glob(os.path.join(BASE, '*/archs.json')):
    for a in json.load(open(f))['archs']:
        arch_of[os.path.join(os.path.dirname(f), a['name'])] = tuple(a['arch'])
# pareto/ (exported by the evolutionary search) has no archs.json: same rule as the export (K evenly spaced by MACs
# rank on the Pareto front); checked against the net.config of dirs whose arch is known
key = lambda d: json.dumps(json.load(open(d + '/net.config'))['blocks'], sort_keys=True)
by_key = {key(d): a for d, a in arch_of.items()}
front = sorted(json.load(open(os.path.join(BASE, 'candidates.json')))['pareto_front'], key=lambda r: r['flops_M'])
pdirs_all = sorted(glob.glob(os.path.join(BASE, 'pareto/arch_*')))
for d, i in zip(pdirs_all, np.linspace(0, len(front) - 1, len(pdirs_all))):
    arch_of[d] = tuple(front[int(round(i))]['arch'])
    if key(d) in by_key:
        assert by_key[key(d)] == arch_of[d], 'pareto export rule mismatch: %s' % d
unique = {}
for d in sorted(arch_of):
    if os.path.exists(d + '/output') and not os.path.exists(d + '/SAME_AS') and arch_of[d] is not None \
            and not os.path.basename(os.path.dirname(d)).startswith('dpp'):
        unique.setdefault(arch_of[d], d)
for d in sorted(glob.glob(os.path.join(BASE, 'pareto/arch_*'))):
    if arch_of[d] is None and os.path.exists(d + '/output'):
        raise RuntimeError('pareto arch not matched: %s' % d)
archs = sorted(unique, key=lambda a: evaluated[a]['flops_M'])
dirs = [unique[a] for a in archs]
flops = np.array([evaluated[a]['flops_M'] for a in archs])
sup = np.array([evaluated[a]['supernet_acc'] for a in archs])
seed_sets = {'large_84.6M': [os.path.join(BASE, 'even6_min76/arch_02')] + [os.path.join(BASE, 'seed_even02/s%d' % k) for k in (1, 2, 3)],
             'small_32.9M': [os.path.join(BASE, 'even10_full/arch_02')] + [os.path.join(BASE, 'seed_even10_02/s%d' % k) for k in (1, 2)]}
print('%d unique architectures (%.1f-%.1fM)' % (len(archs), flops.min(), flops.max()))

# pool similarity S of these archs: cifar10_derive_archs.py --candidates candidates_zero_all.json
# --candidate_set zero_all --select dpp --out_path dpp_zero_all_{feature,prob} (--dpp_diversity)
cand_path = os.path.join(BASE, 'candidates_zero_all.json')
if not os.path.exists(cand_path):
    cand = json.load(open(os.path.join(BASE, 'candidates.json')))
    cand['zero_all'] = [{'arch': list(a), 'flops_M': evaluated[a]['flops_M'], 'supernet_acc': evaluated[a]['supernet_acc']}
                        for a in archs]
    json.dump(cand, open(cand_path, 'w'), indent=1)
    print('wrote %s: run the DPP for S, then rerun' % cand_path)
sims = {}
for tag in ['feature', 'prob']:
    p = os.path.join(BASE, 'dpp_zero_all_%s' % tag, 'archs.json')
    if os.path.exists(p):
        info = json.load(open(p))
        pool = [tuple(a) for a in info['pool_archs']]
        idx = [pool.index(a) for a in archs]
        sims[tag] = np.array(info['dpp']['pool_similarity'])[np.ix_(idx, idx)]

out = {'dirs': [os.path.relpath(d, ROOT) for d in dirs], 'archs': [list(a) for a in archs], 'flops_M': flops.tolist(),
       'supernet_acc': sup.tolist()}
for split in ['valid', 'test']:
    all_dirs = dirs + [d for s in seed_sets.values() for d in s if d not in dirs]
    dataset, vs = run_setting(all_dirs)
    loader = build_loader(split, 500, 4, dataset, vs)
    ds = loader.dataset
    labels = torch.tensor(ds.dataset.targets)[ds.indices] if hasattr(ds, 'indices') else torch.tensor(ds.targets)
    probs = {d: predict(load_net(d, 'last'), loader, device) for d in all_dirs}
    preds = {d: p.argmax(1) for d, p in probs.items()}
    err = {d: (preds[d] != labels).float().mean().item() for d in all_dirs}
    acc = {d: 100 - 100 * err[d] for d in all_dirs}

    def ndis(a, b):
        return (preds[a] != preds[b]).float().mean().item() / (err[a] + err[b])

    def ens(ms):
        m = torch.stack([probs[d] for d in ms]).mean(0)
        return ((m.argmax(1) == labels).float().mean().item() * 100,
                torch.nn.functional.nll_loss(torch.log(m.clamp_min(1e-12)), labels).item())

    r = {}
    real = np.array([acc[d] for d in dirs])
    r['single'] = real.tolist()
    print('== %s' % split)
    for f, s, a, d in zip(flops, sup, real, dirs):
        print('  %6.1fM  supernet %5.2f  retrained %5.2f  %s' % (f, s, a, os.path.relpath(d, BASE)))
    r['kendall_sup'] = kendall(sup, real); r['spearman_sup'] = spearman(sup, real)
    r['kendall_flops'] = kendall(flops, real); r['spearman_flops'] = spearman(flops, real)
    print('  retrained acc vs supernet acc: Spearman %.2f Kendall %.2f | vs MACs: Spearman %.2f Kendall %.2f (%d archs)' % (
        r['spearman_sup'], r['kendall_sup'], r['spearman_flops'], r['kendall_flops'], len(dirs)))
    small = flops < 76
    print('  < 76M (%d archs): Kendall supernet %.2f, MACs %.2f' % (
        small.sum(), kendall(sup[small], real[small]), kendall(flops[small], real[small])))

    pairs = list(itertools.combinations(range(len(dirs)), 2))
    nd = np.array([ndis(dirs[i], dirs[j]) for i, j in pairs])
    dflops = np.array([abs(flops[i] - flops[j]) for i, j in pairs])
    ham = np.array([sum(x != y for x, y in zip(archs[i], archs[j])) for i, j in pairs])
    r['pair_ndis'] = nd.tolist()
    r['spearman_ndis_dflops'] = spearman(dflops, nd)
    r['spearman_ndis_hamming'] = spearman(ham, nd)
    print('  all %d arch pairs: normalized disagreement %.3f +- %.3f | Spearman with |dMACs| %+.2f, Hamming %+.2f' % (
        len(pairs), nd.mean(), nd.std(), r['spearman_ndis_dflops'], r['spearman_ndis_hamming']))
    for tag, S in sims.items():
        r['spearman_ndis_S_' + tag] = spearman([S[i, j] for i, j in pairs], nd)
        print('    Spearman(S %s, ndis) %+.2f' % (tag, r['spearman_ndis_S_' + tag]))
    for lo, hi in [(0, 20), (20, 40), (40, 80)]:
        m = (dflops >= lo) & (dflops < hi)
        print('    |dMACs| %2d-%2dM: %3d pairs, ndis %.3f' % (lo, hi, m.sum(), nd[m].mean()))
    for name, s in seed_sets.items():
        v = [ndis(a, b) for a, b in itertools.combinations(s, 2)]
        r['seed_ndis_' + name] = v
        print('  seed pairs %-12s ndis %.3f +- %.3f (%d pairs), single %s' % (
            name, np.mean(v), np.std(v), len(v), ' '.join('%.2f' % acc[d] for d in s)))
    sm = [i for i in range(len(dirs)) if flops[i] < 42]
    v = [ndis(dirs[i], dirs[j]) for i, j in itertools.combinations(sm, 2)]
    print('  distinct small archs (< 42M, %d): ndis %.3f +- %.3f' % (len(sm), np.mean(v), np.std(v)))

    # ensembles of 3 under ~100M MACs
    rows = []
    for c in itertools.combinations(sm, 3):
        a, nll = ens([dirs[i] for i in c])
        rows.append((flops[list(c)].sum(), a, nll))
    r['ens3_small_distinct'] = rows
    s3 = seed_sets['small_32.9M']
    r['ens3_small_seed'] = ens(s3)
    big = int(np.argmax(flops))
    print('  K=3 distinct small archs: %d sets, MACs %.0f-%.0fM, acc %.2f (%.2f-%.2f), NLL %.4f' % (
        len(rows), min(x[0] for x in rows), max(x[0] for x in rows), np.mean([x[1] for x in rows]),
        min(x[1] for x in rows), max(x[1] for x in rows), np.mean([x[2] for x in rows])))
    print('  K=3 seeds of 32.9M (98.7M): acc %.2f NLL %.4f' % r['ens3_small_seed'])
    print('  single largest (%.1fM): acc %.2f NLL %.4f' % ((flops[big],) + ens([dirs[big]])))
    pareto = sorted(glob.glob(os.path.join(BASE, 'pareto/arch_*')))
    pdirs = [unique[arch_of[d]] for d in pareto]
    pf = sum(evaluated[arch_of[d]]['flops_M'] for d in pareto)
    r['ens5_fairnas_pareto'] = ens(pdirs)
    print('  K=5 FairNAS output pareto/ (%.0fM): acc %.2f NLL %.4f' % ((pf,) + r['ens5_fairnas_pareto']))
    out[split] = r
json.dump(out, open(os.path.join(ROOT, 'Exp6/zero_all_analysis.json'), 'w'), indent=1)
print('Saved to Exp6/zero_all_analysis.json')
