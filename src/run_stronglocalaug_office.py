import sys
import os
sys.path.insert(0, os.getcwd())

from torchvision import transforms as tvt

import data_utils
from common import make_args, augmented_loader, run
from util import prepare_data_office_multi_clients

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
args = make_args("office", seed)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)

strong = tvt.Compose([
    tvt.Resize([72, 72]),
    tvt.RandomResizedCrop(args.size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    tvt.RandomHorizontalFlip(),
    tvt.RandomRotation((-30, 30)),
    tvt.RandomApply([tvt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.8),
    tvt.RandomGrayscale(p=0.1),
    tvt.RandomApply([tvt.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
    tvt.ToTensor(),
])

feature_loader_list = [None] * args.num_users
for i, dom in enumerate(client_domains):
    if dom in {"caltech", "amazon", "dslr"}:
        feature_loader_list[i] = train_loader_list[i]
        train_loader_list[i] = augmented_loader(train_loader_list[i], args, i, data_utils.OfficeDataset,
                                                "../data/office_caltech_10", dom, strong)

run(args, train_loader_list, test_loader_list, client_domains, feature_loader_list,
    lambda_G=0.05, tag="stronglocalaug_office_M4_seed" + str(seed), out_dir="results")
