# Evaluate an ensemble of trained CIFAR-10 networks (outputs of cifar10_run_exp.py)
# Ensemble prediction = average of softmax probabilities over members.

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data_providers.cifar10 import CIFAR10DataProvider
from models.normal_nets.proxyless_nets import ProxylessNASNets
from utils import count_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--paths', type=str, nargs='+', required=True,
                    help='run dirs of cifar10_run_exp.py (each containing net.config)')
parser.add_argument('--weights', type=str, default='last', choices=['last', 'best', 'init'],
                    help='last: checkpoint/checkpoint.pth.tar, best: checkpoint/model_best.pth.tar, init: <path>/init')
parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'],
                    help='valid: the same 5000-image held-out split used for training/search')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--output', type=str, default=None, help='save results as json')


def weight_path(path, weights):
    if weights == 'init':
        return os.path.join(path, 'init')
    name = 'model_best.pth.tar' if weights == 'best' else 'checkpoint.pth.tar'
    return os.path.join(path, 'checkpoint', name)


def load_net(path, weights):
    net = ProxylessNASNets.build_from_config(json.load(open(os.path.join(path, 'net.config'))))
    checkpoint = torch.load(weight_path(path, weights), map_location='cpu')
    net.load_state_dict(checkpoint.get('state_dict', checkpoint))
    return net


def build_loader(split, batch_size, n_worker):
    provider = CIFAR10DataProvider(test_batch_size=batch_size, valid_size=5000 if split == 'valid' else None,
                                   n_worker=n_worker)
    if split == 'test':
        dataset = provider.test.dataset
    else:
        # fixed order so that member predictions are aligned
        dataset = torch.utils.data.Subset(provider.valid.dataset, sorted(provider.valid.sampler.indices))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=n_worker, pin_memory=True)


def predict(net, loader, device):
    net.to(device).eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            probs.append(F.softmax(net(images.to(device, non_blocking=True)), dim=1).cpu())
    net.cpu()
    return torch.cat(probs)


def metrics(probs, labels):
    acc = (probs.argmax(dim=1) == labels).float().mean().item() * 100
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item()
    return acc, nll


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loader = build_loader(args.split, args.batch_size, args.n_worker)
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        labels = torch.tensor(dataset.dataset.targets)[dataset.indices]
    else:
        labels = torch.tensor(dataset.targets)

    members, all_probs = [], []
    for path in args.paths:
        net = load_net(path, args.weights)
        net.eval()
        with torch.no_grad():
            flops, _ = net.get_flops(torch.zeros(1, 3, 32, 32))
        probs = predict(net, loader, device)
        acc, nll = metrics(probs, labels)
        members.append({'path': path, 'acc1': acc, 'nll': nll,
                        'flops_M': flops / 1e6, 'params_M': count_parameters(net) / 1e6})
        all_probs.append(probs)
        print('%s\tacc %.2f\tnll %.4f\tFLOPs %.1fM\tparams %.2fM' % (path, acc, nll, flops / 1e6,
                                                                     members[-1]['params_M']))

    # ensemble of the first m members (m = 1..K), in the given order
    cumulative = []
    for m in range(1, len(all_probs) + 1):
        acc, nll = metrics(torch.stack(all_probs[:m]).mean(dim=0), labels)
        cumulative.append({'n_members': m, 'acc1': acc, 'nll': nll})
    ens_acc, ens_nll = cumulative[-1]['acc1'], cumulative[-1]['nll']

    # diversity: pairwise disagreement of top-1 predictions, oracle = correct by any member
    preds = [p.argmax(dim=1) for p in all_probs]
    disagreement = [[(a != b).float().mean().item() * 100 for b in preds] for a in preds]
    oracle = torch.stack([p == labels for p in preds]).any(dim=0).float().mean().item() * 100
    k = len(preds)
    mean_disagreement = float(np.mean([disagreement[i][j] for i in range(k) for j in range(k) if i < j])) \
        if k > 1 else 0.0

    print('-' * 60)
    for c in cumulative:
        print('ensemble of %d\tacc %.2f\tnll %.4f' % (c['n_members'], c['acc1'], c['nll']))
    print('best single\tacc %.2f' % max(m['acc1'] for m in members))
    print('oracle\t\tacc %.2f' % oracle)
    print('mean pairwise disagreement: %.2f%%' % mean_disagreement)
    print('total FLOPs %.1fM\ttotal params %.2fM' % (sum(m['flops_M'] for m in members),
                                                     sum(m['params_M'] for m in members)))

    if args.output:
        json.dump({
            'split': args.split, 'weights': args.weights, 'members': members,
            'ensemble': {'acc1': ens_acc, 'nll': ens_nll}, 'cumulative': cumulative,
            'oracle_acc1': oracle, 'pairwise_disagreement': disagreement,
            'mean_pairwise_disagreement': mean_disagreement,
            'total_flops_M': sum(m['flops_M'] for m in members),
            'total_params_M': sum(m['params_M'] for m in members),
        }, open(args.output, 'w'), indent=4)
        print('Saved to %s' % args.output)
