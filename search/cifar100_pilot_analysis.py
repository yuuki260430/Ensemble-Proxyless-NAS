# CIFAR-100 pilot: architecture diversity vs seed diversity, and whether the pool similarity S predicts it
# usage: cd search && uv run python cifar100_pilot_analysis.py  (writes Exp6/pilot_analysis.json)
import itertools, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from cifar10_ensemble_eval import load_net, run_setting, build_loader, predict

ROOT = os.path.abspath('..')
SPLITS = ['valid', 'test']
device = torch.device('cuda:0')
out = {}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


for space in ['zero', 'nozero']:
    base = '%s/Exp6/evo_%s' % (ROOT, space)
    archs = ['%s/even6_min76/arch_%02d' % (base, i) for i in range(6)]
    seeds = [archs[2]] + ['%s/seed_even02/s%d' % (base, k) for k in (1, 2, 3)]
    even = json.load(open('%s/even6_min76/archs.json' % base))
    arch_list = [tuple(a['arch']) for a in even['archs']]
    sims = {}
    for tag, d in [('feature', 'dpp6_min76'), ('prob', 'dpp6_min76_prob')]:
        p = '%s/%s/archs.json' % (base, d)
        if os.path.exists(p):
            info = json.load(open(p))
            pool = [tuple(a) for a in info['pool_archs']]
            idx = [pool.index(a) for a in arch_list]
            S = np.array(info['dpp']['pool_similarity'])
            sims[tag] = S[np.ix_(idx, idx)]
    res = {}
    for split in SPLITS:
        dataset, vs = run_setting(archs)
        loader = build_loader(split, 500, 4, dataset, vs)
        ds = loader.dataset
        labels = torch.tensor(ds.dataset.targets)[ds.indices] if hasattr(ds, 'indices') else torch.tensor(ds.targets)
        probs = {d: predict(load_net(d, 'last'), loader, device) for d in archs + seeds[1:]}
        preds = {d: p.argmax(1) for d, p in probs.items()}
        err = {d: (preds[d] != labels).float().mean().item() for d in probs}
        acc = {d: 100 - 100 * err[d] for d in probs}

        def pair(a, b):
            dis = (preds[a] != preds[b]).float().mean().item()
            return dis * 100, dis / (err[a] + err[b])

        def ens(ms):
            m = torch.stack([probs[d] for d in ms]).mean(0)
            return (m.argmax(1) == labels).float().mean().item() * 100

        arch_pairs = list(itertools.combinations(range(6), 2))
        ap = [pair(archs[i], archs[j]) for i, j in arch_pairs]
        sp = [pair(a, b) for a, b in itertools.combinations(seeds, 2)]
        r = {'single_arch': [acc[d] for d in archs], 'single_seed': [acc[d] for d in seeds],
             'arch_dis': [x[0] for x in ap], 'arch_ndis': [x[1] for x in ap],
             'seed_dis': [x[0] for x in sp], 'seed_ndis': [x[1] for x in sp],
             'ens4_seed': ens(seeds),
             'ens4_arch_all': [ens([archs[i] for i in c]) for c in itertools.combinations(range(6), 4)],
             'ens6_arch': ens(archs)}
        for tag, S in sims.items():
            s = [S[i, j] for i, j in arch_pairs]
            r['spearman_S_vs_ndis_' + tag] = spearman(s, r['arch_ndis'])
            r['spearman_S_vs_dis_' + tag] = spearman(s, r['arch_dis'])
        res[split] = r
        print('== %s %s' % (space, split))
        print('  single acc archs %s' % ' '.join('%.2f' % a for a in r['single_arch']))
        print('  single acc seeds %s' % ' '.join('%.2f' % a for a in r['single_seed']))
        print('  disagreement  arch pairs %.2f +- %.2f (%.2f-%.2f) | seed pairs %.2f +- %.2f (%.2f-%.2f)' % (
            np.mean(r['arch_dis']), np.std(r['arch_dis']), min(r['arch_dis']), max(r['arch_dis']),
            np.mean(r['seed_dis']), np.std(r['seed_dis']), min(r['seed_dis']), max(r['seed_dis'])))
        print('  normalized    arch pairs %.3f +- %.3f | seed pairs %.3f +- %.3f' % (
            np.mean(r['arch_ndis']), np.std(r['arch_ndis']), np.mean(r['seed_ndis']), np.std(r['seed_ndis'])))
        e = r['ens4_arch_all']
        print('  K=4 ensemble  seed %.2f | arch subsets mean %.2f (%.2f-%.2f) | K=6 archs %.2f' % (
            r['ens4_seed'], np.mean(e), min(e), max(e), r['ens6_arch']))
        for k, v in r.items():
            if k.startswith('spearman'):
                print('  %s %+.2f' % (k, v))
    out[space] = res
json.dump(out, open(os.path.join(ROOT, 'Exp6/pilot_analysis.json'), 'w'), indent=1)
