import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

from option import args_parser
from federated_main import ours, set_seed


class PersistentRNGCompose:
    def __init__(self, transform, seed):
        self.transform = transform
        self._python_state = random.Random(seed).getstate()
        self._numpy_state = np.random.RandomState(seed).get_state()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self._torch_state = gen.get_state().clone()

    def __call__(self, img):
        caller = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        try:
            random.setstate(self._python_state)
            np.random.set_state(self._numpy_state)
            torch.set_rng_state(self._torch_state)
            out = self.transform(img)
            self._python_state = random.getstate()
            self._numpy_state = np.random.get_state()
            self._torch_state = torch.get_rng_state().clone()
            return out
        finally:
            random.setstate(caller[0])
            np.random.set_state(caller[1])
            torch.set_rng_state(caller[2])


def make_args(dataset, seed):
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 100
    args.lr = 0.01
    args.number_workers = 0
    args.dataset = dataset
    args.num_classes = 10 if dataset == "office" else 7
    args.size = 64
    args.batch = 32
    args.wk_iters = 10
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.adcol_epoch = 3
    args.device = args.device if torch.cuda.is_available() else "cpu"
    args.domain_keyed_proto = True
    args.no_discriminator_fix = False
    args.seed = seed
    set_seed(args)
    return args


def resampled_loader(loader, args, client_idx):
    ds = loader.dataset
    n = len(loader.sampler.indices)
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + client_idx)
    return DataLoader(ds, batch_size=args.batch,
                      sampler=RandomSampler(ds, replacement=False, num_samples=n, generator=g),
                      num_workers=args.number_workers, pin_memory=True)


def augmented_loader(loader, args, client_idx, dataset_cls, base_path, domain, transform):
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + client_idx)
    ds = dataset_cls(base_path, domain,
                     transform=PersistentRNGCompose(transform, seed=args.seed + 30000 + client_idx))
    return DataLoader(ds, batch_size=args.batch,
                      sampler=SubsetRandomSampler(loader.sampler.indices, generator=g),
                      num_workers=args.number_workers, pin_memory=True)


def run(args, train_loader_list, test_loader_list, client_domains, feature_loader_list, lambda_G, tag, out_dir):
    out_dir = os.path.join(out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    acc_log_path = os.path.join(out_dir, tag + "_acc.csv")
    if os.path.exists(acc_log_path):
        raise RuntimeError(acc_log_path + " already exists")
    _, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        acc_log_path=acc_log_path, weights_dir=os.path.join(out_dir, "weights") + os.sep,
        lambda_G=lambda_G, feature_loader_list=feature_loader_list,
        preserve_feature_probe_state=True,
    )
    for d in datasets_name:
        a = accuracy_list[d]
        print(f"{d:<12} final={a[-1]:.4f} peak={max(a):.4f}@r{a.index(max(a))}")
    return accuracy_list, datasets_name
