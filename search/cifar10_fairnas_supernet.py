# FairNAS baseline, stage 1: supernet training with strict fairness (CIFAR-10)
# X. Chu, B. Zhang, R. Xu. "FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture
# Search." ICCV 2021 (arXiv:1907.01845), Algorithm 1.
#
# Every step builds m single-path models that use each candidate op of every layer exactly once
# (layers with fewer than m candidates are padded with randomly chosen ops, Appendix A.1), back-propagates
# each of them on the same mini-batch and updates the supernet once. No architecture parameters are learned;
# stage 2 (cifar10_fairnas_evo.py) searches with the trained supernet as the evaluator.
#
# Defaults follow the paper (Sec. 4.1, search space (a)/(b)): 150 epochs, batch 256, SGD momentum 0.9,
# lr 0.045 with cosine decay, weight decay 4e-5, gradients summed over the m paths.
#
# Output: <path>/checkpoint/warmup.pth.tar (supernet weights) and <path>/supernet.config (net settings
# used by stage 2 and cifar10_derive_archs.py).

import argparse

from models import CIFAR10RunConfig
from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--manual_seed', type=int, default=0)

""" supernet training (FairNAS Sec. 4.1) """
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--init_lr', type=float, default=0.045)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=5000,
                    help='held-out part of the CIFAR-10 train set (same split as the ProxylessNAS runs)')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_true', help='the paper only states momentum 0.9')
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.0, help='not used for the FairNAS supernet')
parser.add_argument('--fair_grad', type=str, default='sum', choices=['sum', 'mean'],
                    help='sum: FairNAS Algorithm 1; mean: divide by the number of paths using each parameter')
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--print_frequency', type=int, default=10)

""" search space """
parser.add_argument('--no_zero', action='store_true',
                    help='FairNAS search space: 6 MBConv ops per layer without the Zero (skip) op')
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='1,2,2,1,2,1')
parser.add_argument('--first_stride', type=int, default=1)
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.0)

CONV_CANDIDATES = [
    '3x3_MBConv3', '3x3_MBConv6',
    '5x5_MBConv3', '5x5_MBConv6',
    '7x7_MBConv3', '7x7_MBConv6',
]


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.path, exist_ok=True)

    run_config = CIFAR10RunConfig(
        n_epochs=args.n_epochs, init_lr=args.init_lr, lr_schedule_type='cosine', dataset='cifar10',
        train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size, valid_size=args.valid_size,
        opt_type='sgd', opt_param={'momentum': args.momentum, 'nesterov': args.nesterov},
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing, no_decay_keys=None,
        print_frequency=args.print_frequency, n_worker=args.n_worker,
    )

    supernet_config = {
        'width_stages': args.width_stages, 'n_cell_stages': args.n_cell_stages,
        'stride_stages': args.stride_stages, 'first_stride': args.first_stride, 'width_mult': args.width_mult,
        'bn_momentum': args.bn_momentum, 'bn_eps': args.bn_eps, 'dropout': args.dropout,
        'conv_candidates': CONV_CANDIDATES, 'allow_zero': not args.no_zero,
    }
    super_net = SuperProxylessNASNets(
        width_stages=[int(v) for v in args.width_stages.split(',')],
        n_cell_stages=[int(v) for v in args.n_cell_stages.split(',')],
        stride_stages=[int(v) for v in args.stride_stages.split(',')],
        conv_candidates=CONV_CANDIDATES, n_classes=run_config.data_provider.n_classes,
        width_mult=args.width_mult, bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout,
        first_stride=args.first_stride, allow_zero=not args.no_zero,
    )
    n_choices = [e.n_choices for e in super_net.redundant_modules]
    print('searchable layers: %d\tchoices per layer: %s' % (len(n_choices), n_choices))

    # the architecture parameters built by ArchSearchRunManager are never used or updated here
    arch_search_config = GradientArchSearchConfig()
    manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)

    json.dump(dict(supernet_config, training={**vars(args)}), open(os.path.join(args.path, 'supernet.config'), 'w'),
              indent=4)
    json.dump(run_config.config, open(os.path.join(args.path, 'run.config'), 'w'), indent=4)

    if args.resume:
        manager.load_model(model_fname=os.path.join(manager.run_manager.save_path, 'warmup.pth.tar'))
    if manager.warmup:
        manager.warm_up(warmup_epochs=args.n_epochs, fair=True, lr_max=args.init_lr, fair_grad=args.fair_grad)
    print('Supernet training finished: %s' % os.path.join(manager.run_manager.save_path, 'warmup.pth.tar'))
