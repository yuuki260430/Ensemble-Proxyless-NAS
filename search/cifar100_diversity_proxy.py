# Which pre-retraining (supernet) diversity measure predicts the retrained prediction diversity, beyond size / quality?
# For the retrained archs of Exp6/zero_all_analysis.json: subnet predictions with inherited weights (BN recalibrated as
# in the evolutionary search) on the valid split, pairwise measures, and their (partial) Spearman correlation with the
# retrained normalized disagreement. Controls: |dMACs|, |d supernet acc|, min supernet acc of the pair.
# usage: cd search && uv run python cifar100_diversity_proxy.py  (writes Exp6/diversity_proxy.json)
import argparse
import itertools
import json
import os

import numpy as np
import torch

from cifar10_fairnas_evo import build_super_net, SupernetEvaluator
from modules.mix_op import MixedEdge

ROOT = os.path.abspath('..')
BASE = os.path.join(ROOT, 'Exp6/evo_zero')
device = torch.device('cuda:0')


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial(y, x, ctrl):
    C = np.column_stack([np.ones(len(y))] + [rank(c) for c in ctrl])
    ry = rank(y) - C @ np.linalg.lstsq(C, rank(y), rcond=None)[0]
    rx = rank(x) - C @ np.linalg.lstsq(C, rank(x), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def subnet_probs(ev, arch):
    """ same as SupernetEvaluator.accuracy, but returns the softmax outputs on the valid split """
    MixedEdge.MODE = None
    for e, i in zip(ev.edges, arch):
        e.active_index = [i]
        e.inactive_index = [j for j in range(e.n_choices) if j != i]
    ev.net.unused_modules_off()
    for m in ev.net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    ev.net.train()
    for x in ev.bn_batches:
        ev.net(x)
    ev.net.eval()
    p = torch.cat([torch.softmax(ev.net(x), 1) for x in ev.valid_x]).cpu()
    ev.net.unused_modules_back()
    return p


if __name__ == '__main__':
    res = json.load(open(os.path.join(ROOT, 'Exp6/zero_all_analysis.json')))
    archs = [tuple(a) for a in res['archs']]
    flops = np.array(res['flops_M'])
    cand = json.load(open(os.path.join(BASE, 'candidates.json')))
    cfg = json.load(open(os.path.join(cand['supernet_path'], 'supernet.config')))
    net = build_super_net(cfg)
    sd = net.state_dict()
    sd.update(torch.load(cand['checkpoint'], map_location='cpu')['state_dict'])
    net.load_state_dict(sd)
    args = argparse.Namespace(dataset=cfg.get('dataset', 'cifar10'), batch_size=500, valid_size=5000, n_worker=4,
                              manual_seed=0, bn_images=5000)
    ev = SupernetEvaluator(net, args, device)
    labels = torch.cat(ev.valid_y).cpu()
    probs = [subnet_probs(ev, a) for a in archs]
    preds = [p.argmax(1) for p in probs]
    wrong = [(p != labels) for p in preds]
    err = np.array([w.float().mean().item() for w in wrong])
    sup = 100 * (1 - err)  # supernet acc measured here (same protocol as the search)

    pairs = list(itertools.combinations(range(len(archs)), 2))
    m = {}
    for i, j in pairs:
        dis = (preds[i] != preds[j]).float().mean().item()
        wi, wj = wrong[i].float(), wrong[j].float()
        both = (wi * wj).mean().item()
        phi = float(np.corrcoef(wi.numpy(), wj.numpy())[0, 1])  # correlation of the error indicators
        vals = {
            'S_prob (current, 1 - similarity)': None,
            'disagreement': dis,
            'normalized disagreement d/(e_i+e_j)': dis / (err[i] + err[j]),
            'error correlation (phi), negated': -phi,
            'double fault / min(e_i,e_j), negated': -both / min(err[i], err[j]),
            'prob L2 distance (DivBO Eq.3)': (probs[i] - probs[j]).norm(dim=1).mean().item(),
            'disagreement on samples both wrong': ((preds[i] != preds[j]) & wrong[i] & wrong[j]).float().sum().item()
                                                  / max(1.0, (wrong[i] & wrong[j]).float().sum().item()),
        }
        for k, v in vals.items():
            m.setdefault(k, []).append(v)
    info = json.load(open(os.path.join(BASE, 'dpp_zero_all_prob', 'archs.json')))
    pool = [tuple(a) for a in info['pool_archs']]
    idx = [pool.index(a) for a in archs]
    S = np.array(info['dpp']['pool_similarity'])[np.ix_(idx, idx)]
    m['S_prob (current, 1 - similarity)'] = [1 - S[i, j] for i, j in pairs]

    dF = [abs(flops[i] - flops[j]) for i, j in pairs]
    dQ = [abs(sup[i] - sup[j]) for i, j in pairs]
    mQ = [min(sup[i], sup[j]) for i, j in pairs]
    out = {'archs': res['archs'], 'supernet_acc_here': sup.tolist(), 'measures': m}
    print('%d archs, %d pairs; supernet acc %.1f-%.1f%%' % (len(archs), len(pairs), sup.min(), sup.max()))
    print('%-42s %s' % ('pre-retraining measure (larger = more diverse)',
                        '  raw(valid/test)   partial: ctrl size&quality (valid/test)'))
    for split in ['valid', 'test']:
        target = res[split]['pair_ndis']
        out[split] = {k: {'raw': spearman(v, target), 'partial': partial(target, v, [dF, dQ, mQ])} for k, v in m.items()}
    for k in m:
        print('%-42s   %+.2f / %+.2f        %+.2f / %+.2f' % (
            k, out['valid'][k]['raw'], out['test'][k]['raw'], out['valid'][k]['partial'], out['test'][k]['partial']))
    for k, c in [('min supernet acc (negated)', [-x for x in mQ]), ('|d supernet acc|', dQ), ('|dMACs|', dF)]:
        print('%-42s   %+.2f / %+.2f' % (k, spearman(c, res['valid']['pair_ndis']), spearman(c, res['test']['pair_ndis'])))
    json.dump(out, open(os.path.join(ROOT, 'Exp6/diversity_proxy.json'), 'w'), indent=1)
