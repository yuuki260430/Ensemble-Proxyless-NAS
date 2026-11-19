# ProxylessNAS: CIFAR-10 / CIFAR-100 Data Providers

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *


class CIFAR10DataProvider(DataProvider):
    DATASET = datasets.CIFAR10
    NAME = 'cifar10'
    N_CLASSES = 10
    URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  # values used by all CIFAR-10 runs so far

    def __init__(self, save_path=None, train_batch_size=128, test_batch_size=128, valid_size=None,
                 n_worker=4, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        train_dataset = self.DATASET(
            root=self.save_path, train=True, transform=train_transforms, download=True
        )

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            train_indexes, valid_indexes = self.random_sample_valid_set(
                train_dataset.targets, valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            valid_dataset = self.DATASET(
                root=self.save_path, train=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize,
                ]), download=True
            )

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
                drop_last=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
                drop_last=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            self.DATASET(root=self.save_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ]), download=True),
            batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @classmethod
    def name(cls):
        return cls.NAME

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return self.N_CLASSES

    @property
    def save_path(self):
        if self._save_path is None:
            # Store in project directory (will be ignored by git)
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self._save_path = os.path.join(project_root, 'data', self.NAME)
            os.makedirs(self._save_path, exist_ok=True)
        return self._save_path

    @property
    def data_url(self):
        return self.URL

    @property
    def normalize(self):
        return transforms.Normalize(mean=self.MEAN, std=self.STD)

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def image_size(self):
        return 32


class CIFAR100DataProvider(CIFAR10DataProvider):
    """ same pipeline as CIFAR-10 (split, augmentation); only the dataset and the normalization differ """
    DATASET = datasets.CIFAR100
    NAME = 'cifar100'
    N_CLASSES = 100
    URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    MEAN, STD = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]


CIFAR_PROVIDERS = {'cifar10': CIFAR10DataProvider, 'cifar100': CIFAR100DataProvider}


def get_cifar_provider(dataset):
    """ data provider class for 'cifar10' / 'cifar100' """
    if dataset not in CIFAR_PROVIDERS:
        raise ValueError('do not support: %s' % dataset)
    return CIFAR_PROVIDERS[dataset]
