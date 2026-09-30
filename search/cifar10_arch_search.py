# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Extended for CIFAR-10 dataset

import argparse

from models import CIFAR10RunConfig
from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets
from utils.config_loader import ConfigLoader

# ref values for CIFAR-10 (32x32 input, first_stride=1, stride_stages=1,2,2,1,2,1)
# ImageNet ref FLOPs x 0.30: proxyless_mobile/gpu/cpu measured at 0.28-0.30x of their ImageNet FLOPs
# under this CIFAR-10 configuration.
# The mobile latency lookup table only covers 224x224 inputs, so only 'flops' is supported.
ref_values = {
    'flops': {
        '0.35': 18 * 1e6,
        '0.50': 29 * 1e6,
        '0.75': 63 * 1e6,
        '1.00': 90 * 1e6,
        '1.30': 153 * 1e6,
        '1.40': 175 * 1e6,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--init_lr', type=float, default=0.05)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=5000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='1,2,2,1,2,1')
parser.add_argument('--first_stride', type=int, default=1)
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'])
parser.add_argument('--warmup_epochs', type=int, default=20)
parser.add_argument('--warmup_fair', action='store_true', help='FairNAS strict fairness during warmup')
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1', type=float, default=0.0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None, choices=['flops', None])
parser.add_argument('--hw-config', type=str, default=None, help='Hardware configuration YAML file path (e.g., configs/config_5080.yaml)')
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full_v2', 'full', 'two'])
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='mul#log', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = CIFAR10RunConfig(
        **args.__dict__
    )

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 256
        run_config.test_batch_size = 256
        run_config.valid_size = 256
        run_config.n_worker = 0

    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
    super_net = SuperProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, first_stride=args.first_stride
    )

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    if args.target_hardware is None:
        args.ref_value = None
    else:
        args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig
        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    elif args.arch_algo == 'rl':
        from nas_manager import RLArchSearchConfig
        arch_search_config = RLArchSearchConfig(**args.__dict__)
    else:
        raise NotImplementedError

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # Load hardware configuration if specified
    latency_estimator_url = None
    if args.hw_config:
        config = ConfigLoader.load_config(args.hw_config)
        latency_estimator_url = ConfigLoader.get_latency_model_url(config, args.target_hardware)
        print(f"Hardware config loaded from {args.hw_config}")
        if latency_estimator_url:
            print(f"Latency model URL: {latency_estimator_url}")

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config, 
                                                    latency_estimator_url=latency_estimator_url)

    # resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception:
            from pathlib import Path
            home = str(Path.home())
            warmup_path = os.path.join(
                home, 'Workspace/Exp/arch_search/%s_ProxylessNAS_%.2f_%s/warmup.pth.tar' %
                      (run_config.dataset, args.width_mult, width_stages_str)
            )
            if os.path.exists(warmup_path):
                print('load warmup weights')
                arch_search_run_manager.load_model(model_fname=warmup_path)
            else:
                print('fail to load models')

    # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs, fair=args.warmup_fair)

    # joint training
    arch_search_run_manager.train(fix_net_weights=args.debug)
