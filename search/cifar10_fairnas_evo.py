# FairNAS baseline, stage 2: multi-objective evolutionary search with the fair supernet (CIFAR-10)
# X. Chu, B. Zhang, R. Xu. "FairNAS" (ICCV 2021, arXiv:1907.01845), Algorithm 2 and Appendix C.2.
#
#   objectives : supernet accuracy on the held-out split (max), multiply-adds (min), #params (min)  (Sec. 5.2)
#   search     : NSGA-II (Deb et al., 2002). Uniform initialization of P_1 and Q_1; each generation evaluates
#                R_i = P_i U Q_i with inherited supernet weights, keeps N individuals by non-dominated rank and
#                crowding distance, and builds Q_{i+1} by binary tournament + crossover / mutation.
#   setting    : population N = 64, 200 generations (12,800 sampled models in the paper), mutation ratio 0.8
#   output     : the Pareto front of the last population P_{G+1}; K evenly spaced models are exported
#
# Differences from the paper (state them when reporting this baseline):
#   * The MoreMNAS hierarchical mutation (random 0.2 / PPO-reinforced 0.65 / prior regulator 0.15) is replaced by
#     random mutation only: no PPO controller (the official search code is not public).
#   * Each subnet is evaluated after recomputing its BN statistics on a fixed set of training images
#     (as in SPOS; FairNAS reports the same ranking with or without it, Table 6).

import argparse
import copy
import json
import os
import pickle
import shutil
import time

import numpy as np
import torch

from data_providers.cifar10 import get_cifar_provider
from models.super_nets.super_proxyless import SuperProxylessNASNets
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge
from utils import count_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--supernet_path', type=str, required=True, help='output dir of cifar10_fairnas_supernet.py')
parser.add_argument('--checkpoint', type=str, default=None, help='default: <supernet_path>/checkpoint/warmup.pth.tar')
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', action='store_true')

""" NSGA-II (FairNAS Table 8) """
parser.add_argument('--population', type=int, default=64)
parser.add_argument('--generations', type=int, default=200)
parser.add_argument('--mutation_ratio', type=float, default=0.8,
                    help='fraction of offspring made by mutation; the rest by single-point crossover')
parser.add_argument('--mutation_prob', type=float, default=None, help='per-layer mutation probability (default 1/L)')

""" subnet evaluation """
parser.add_argument('--valid_size', type=int, default=5000, help='must match the supernet training split')
parser.add_argument('--bn_images', type=int, default=5000, help='training images for BN recalibration')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_worker', type=int, default=4)

""" export """
parser.add_argument('--n_archs', type=int, default=5, help='K evenly spaced models (by MACs) from the Pareto front')
parser.add_argument('--run_config', type=str, required=True,
                    help='run.config for retraining the exported models (use the same one as the other baselines)')
parser.add_argument('--dropout', type=float, default=None,
                    help='classifier dropout of the exported net.config (the ProxylessNAS baselines used 0.3); '
                         'default: the supernet value')


def build_super_net(cfg):
    return SuperProxylessNASNets(
        width_stages=[int(v) for v in cfg['width_stages'].split(',')],
        n_cell_stages=[int(v) for v in cfg['n_cell_stages'].split(',')],
        stride_stages=[int(v) for v in cfg['stride_stages'].split(',')],
        conv_candidates=list(cfg['conv_candidates']),
        n_classes=get_cifar_provider(cfg.get('dataset', 'cifar10')).N_CLASSES, width_mult=cfg['width_mult'],
        bn_param=(cfg['bn_momentum'], cfg['bn_eps']), dropout_rate=cfg['dropout'],
        first_stride=cfg['first_stride'], allow_zero=cfg['allow_zero'],
    )


def to_normal_net(super_net, arch):
    """ replace each MixedEdge with the candidate op given by `arch` (one index per MixedEdge) """
    net = copy.deepcopy(super_net)
    edges = iter(arch)
    for block in net.blocks:
        if isinstance(block.mobile_inverted_conv, MixedEdge):
            block.mobile_inverted_conv = block.mobile_inverted_conv.candidate_ops[next(edges)]
    return ProxylessNASNets(net.first_conv, list(net.blocks), net.feature_mix_layer, net.classifier)


class CostTable:
    """
    MACs / #params of an architecture = fixed part + sum of the chosen op of every layer. The input shape of a
    layer does not depend on the choices (Zero keeps the shape), so each op is counted once.
    """

    def __init__(self, super_net):
        net = copy.deepcopy(super_net).cpu().eval()
        self.op_flops, self.op_params = [], []
        x = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            _, x = net.first_conv.get_flops(x)
            for block in net.blocks:
                conv = block.mobile_inverted_conv
                if isinstance(conv, MixedEdge):
                    self.op_flops.append([op.get_flops(x)[0] for op in conv.candidate_ops])
                    self.op_params.append([count_parameters(op) for op in conv.candidate_ops])
                    x = block.shortcut(x) if block.shortcut is not None else conv.candidate_ops[0](x)
                else:
                    x = block(x)
        ref = tuple(0 for _ in self.op_flops)
        ref_net = to_normal_net(super_net, ref)
        self.base_flops = self._net_flops(ref_net) - self._sum(self.op_flops, ref)
        self.base_params = count_parameters(ref_net) - self._sum(self.op_params, ref)

    @staticmethod
    def _sum(table, arch):
        return sum(t[i] for t, i in zip(table, arch))

    @staticmethod
    def _net_flops(net):
        net.eval()
        with torch.no_grad():
            return net.get_flops(torch.zeros(1, 3, 32, 32))[0]

    def flops(self, arch):
        return self.base_flops + self._sum(self.op_flops, arch)

    def params(self, arch):
        return self.base_params + self._sum(self.op_params, arch)

    def check(self, super_net, archs):
        for arch in archs:
            net = to_normal_net(super_net, arch)
            assert abs(self._net_flops(net) - self.flops(arch)) < 1, (arch, self._net_flops(net), self.flops(arch))
            assert count_parameters(net) == self.params(arch), (arch, count_parameters(net), self.params(arch))


class SupernetEvaluator:
    """ accuracy of a subnet with inherited weights: BN recalibration on fixed train images, then the valid split """

    def __init__(self, super_net, args, device):
        self.net = super_net.to(device)
        self.edges = self.net.redundant_modules
        self.device = device
        provider = get_cifar_provider(args.dataset)(train_batch_size=args.batch_size,
                                                    test_batch_size=args.batch_size,
                                                    valid_size=args.valid_size, n_worker=args.n_worker)
        g = torch.Generator()
        g.manual_seed(args.manual_seed)
        bn_idx = torch.tensor(provider.train.sampler.indices)[torch.randperm(len(provider.train.sampler.indices),
                                                                             generator=g)[:args.bn_images]]
        # train-time augmentation applied once, so every subnet sees exactly the same calibration images
        bn_set = torch.utils.data.Subset(provider.train.dataset, bn_idx.tolist())
        self.bn_batches = [x.to(device) for x, _ in torch.utils.data.DataLoader(
            bn_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)]
        valid = [(x.to(device), y.to(device)) for x, y in provider.valid]
        self.valid_x = [x for x, _ in valid]
        self.valid_y = [y for _, y in valid]
        self.n_valid = sum(len(y) for y in self.valid_y)

    @torch.no_grad()
    def accuracy(self, arch):
        MixedEdge.MODE = None
        for e, i in zip(self.edges, arch):
            e.active_index = [i]
            e.inactive_index = [j for j in range(e.n_choices) if j != i]
        self.net.unused_modules_off()
        for m in self.net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None  # cumulative average over the calibration batches
        self.net.train()
        for x in self.bn_batches:
            self.net(x)
        self.net.eval()
        correct = sum((self.net(x).argmax(1) == y).sum().item() for x, y in zip(self.valid_x, self.valid_y))
        self.net.unused_modules_back()
        return 100.0 * correct / self.n_valid


""" NSGA-II """


def dominates(a, b):
    """ objectives are minimized """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(objs):
    n = len(objs)
    dominated_by = [[] for _ in range(n)]
    n_dominating = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if dominates(objs[p], objs[q]):
                dominated_by[p].append(q)
            elif dominates(objs[q], objs[p]):
                n_dominating[p] += 1
        if n_dominating[p] == 0:
            fronts[0].append(p)
    while fronts[-1]:
        nxt = []
        for p in fronts[-1]:
            for q in dominated_by[p]:
                n_dominating[q] -= 1
                if n_dominating[q] == 0:
                    nxt.append(q)
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(objs, front):
    dist = {i: 0.0 for i in front}
    for k in range(len(objs[front[0]])):
        order = sorted(front, key=lambda i: objs[i][k])
        lo, hi = objs[order[0]][k], objs[order[-1]][k]
        dist[order[0]] = dist[order[-1]] = float('inf')
        if hi == lo:
            continue
        for a, i, b in zip(order, order[1:], order[2:]):
            dist[i] += (objs[b][k] - objs[a][k]) / (hi - lo)
    return dist


def environmental_selection(pop, objs, n):
    """ keep n individuals by rank, then crowding distance; returns (selected, rank, crowding) """
    fronts = non_dominated_sort(objs)
    chosen, rank, crowd = [], {}, {}
    for r, front in enumerate(fronts):
        dist = crowding_distance(objs, front)
        for i in front:
            rank[i], crowd[i] = r, dist[i]
        if len(chosen) + len(front) <= n:
            chosen += front
        else:
            chosen += sorted(front, key=lambda i: -dist[i])[:n - len(chosen)]
            break
    return [pop[i] for i in chosen], [rank[i] for i in chosen], [crowd[i] for i in chosen]


def tournament(rng, rank, crowd):
    a, b = rng.randint(len(rank), size=2)
    if rank[a] != rank[b]:
        return a if rank[a] < rank[b] else b
    return a if crowd[a] >= crowd[b] else b


def make_offspring(rng, parents, rank, crowd, n_choices, n, mutation_ratio, mutation_prob, exclude):
    children = []
    tries = 0
    while len(children) < n:
        tries += 1
        p1 = parents[tournament(rng, rank, crowd)]
        if rng.rand() < mutation_ratio:
            child = list(p1)
            mutated = [l for l in range(len(child)) if rng.rand() < mutation_prob]
            if not mutated:
                mutated = [rng.randint(len(child))]
            for l in mutated:
                child[l] = int(rng.choice([c for c in range(n_choices[l]) if c != child[l]]))
        else:
            p2 = parents[tournament(rng, rank, crowd)]
            cut = rng.randint(1, len(p1))
            child = list(p1[:cut]) + list(p2[cut:])
        child = tuple(child)
        if child in exclude and tries < 100 * n:
            continue
        exclude.add(child)
        children.append(child)
    return children


def random_arch(rng, n_choices):
    return tuple(int(rng.randint(c)) for c in n_choices)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    os.makedirs(args.out_path, exist_ok=True)

    cfg = json.load(open(os.path.join(args.supernet_path, 'supernet.config')))
    args.dataset = cfg.get('dataset', 'cifar10')  # supernets trained before CIFAR-100 support have no key
    assert json.load(open(args.run_config)).get('dataset', 'cifar10') == args.dataset, 'run_config dataset mismatch'
    ckpt_path = args.checkpoint or os.path.join(args.supernet_path, 'checkpoint', 'warmup.pth.tar')
    super_net = build_super_net(cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if checkpoint.get('warmup', False):
        print('WARNING: supernet training is not finished (warmup_epoch %s)' % checkpoint.get('warmup_epoch'))
    model_dict = super_net.state_dict()
    model_dict.update(checkpoint['state_dict'])  # architecture parameters are not saved (unused here)
    super_net.load_state_dict(model_dict)
    print('Loaded supernet from %s' % ckpt_path)

    n_choices = [e.n_choices for e in super_net.redundant_modules]
    op_names = [[op.module_str for op in e.candidate_ops] for e in super_net.redundant_modules]
    L = len(n_choices)
    mutation_prob = args.mutation_prob if args.mutation_prob is not None else 1.0 / L
    rng = np.random.RandomState(args.manual_seed)

    costs = CostTable(super_net)
    costs.check(super_net, [random_arch(np.random.RandomState(s), n_choices) for s in range(5)])
    print('MACs / params table verified against full networks')

    evaluator = SupernetEvaluator(super_net, args, device)

    state_path = os.path.join(args.out_path, 'evo_state.pkl')
    if args.resume and os.path.exists(state_path):
        state = pickle.load(open(state_path, 'rb'))
        cache, P, Q, start_gen = state['cache'], state['P'], state['Q'], state['generation']
        rng.set_state(state['rng'])
        print('Resumed at generation %d (%d evaluated)' % (start_gen, len(cache)))
    else:
        cache = {}
        seen = set()
        P = []
        while len(P) < 2 * args.population:
            a = random_arch(rng, n_choices)
            if a not in seen:
                seen.add(a)
                P.append(a)
        P, Q = P[:args.population], P[args.population:]
        start_gen = 0

    def evaluate(arch):
        if arch not in cache:
            cache[arch] = {'acc': evaluator.accuracy(arch), 'flops': costs.flops(arch), 'params': costs.params(arch)}
        c = cache[arch]
        return (-c['acc'], c['flops'], c['params'])

    log = open(os.path.join(args.out_path, 'evo.log'), 'a')
    for gen in range(start_gen, args.generations):
        t0 = time.time()
        R = list(dict.fromkeys(P + Q))  # union without duplicates
        objs = [evaluate(a) for a in R]
        P, rank, crowd = environmental_selection(R, objs, args.population)
        front0 = [a for a, r in zip(P, rank) if r == 0]
        Q = make_offspring(rng, P, rank, crowd, n_choices, args.population, args.mutation_ratio, mutation_prob,
                           exclude=set(P))
        accs = [cache[a]['acc'] for a in P]
        flops = [cache[a]['flops'] / 1e6 for a in P]
        msg = ('gen %3d/%d\tevaluated %5d\tfront %2d\tacc max %.2f mean %.2f\tMACs %.1f-%.1fM\t%.1fs' % (
            gen + 1, args.generations, len(cache), len(front0), max(accs), np.mean(accs), min(flops), max(flops),
            time.time() - t0))
        print(msg)
        log.write(msg + '\n')
        log.flush()
        pickle.dump({'cache': cache, 'P': P, 'Q': Q, 'generation': gen + 1, 'rng': rng.get_state()},
                    open(state_path + '.tmp', 'wb'))
        os.replace(state_path + '.tmp', state_path)

    # final population P_{G+1} (its members are already evaluated) and its Pareto front
    objs = [evaluate(a) for a in P]
    front = [P[i] for i in non_dominated_sort(objs)[0]]
    front = sorted(front, key=lambda a: cache[a]['flops'])

    def describe(a):
        c = cache[a]
        return {'arch': list(a), 'ops': [op_names[l][i] for l, i in enumerate(a)], 'supernet_acc': c['acc'],
                'flops_M': round(c['flops'] / 1e6, 2), 'params_M': round(c['params'] / 1e6, 3)}

    json.dump({
        'supernet_path': args.supernet_path, 'checkpoint': ckpt_path, 'args': vars(args),
        'n_evaluated': len(cache), 'n_choices': n_choices,
        'pareto_front': [describe(a) for a in front],
        'final_population': [describe(a) for a in P],
    }, open(os.path.join(args.out_path, 'candidates.json'), 'w'), indent=4)
    json.dump([describe(a) for a in cache], open(os.path.join(args.out_path, 'all_evaluated.json'), 'w'))

    # K evenly spaced models (by MACs) on the front, exported for scratch retraining (no `init` file)
    K = min(args.n_archs, len(front))
    picks = [front[int(round(i))] for i in np.linspace(0, len(front) - 1, K)]
    super_net = super_net.cpu()
    # BN recalibration set momentum = None (cumulative average); restore it, or net.config stores null and
    # retraining would use a cumulative running average over the whole training run
    super_net.set_bn_param(momentum=cfg['bn_momentum'], eps=cfg['bn_eps'])
    for idx, a in enumerate(picks):
        arch_dir = os.path.join(args.out_path, 'pareto', 'arch_%02d' % idx)
        os.makedirs(arch_dir, exist_ok=True)
        net_config = to_normal_net(super_net, a).config
        if args.dropout is not None:
            net_config['classifier']['dropout_rate'] = args.dropout
        json.dump(net_config, open(os.path.join(arch_dir, 'net.config'), 'w'), indent=4)
        shutil.copy(args.run_config, os.path.join(arch_dir, 'run.config'))
        d = describe(a)
        print('arch_%02d\tsupernet acc %.2f\tMACs %.1fM\tparams %.2fM' % (
            idx, d['supernet_acc'], d['flops_M'], d['params_M']))
    print('Pareto front: %d models, exported %d to %s' % (len(front), K, os.path.join(args.out_path, 'pareto')))
