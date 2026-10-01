import sys
import os
sys.path.insert(0, os.getcwd())

from common import make_args, resampled_loader, run
from util import prepare_data_office_multi_clients

ARMS = ("local", "server", "both")
if len(sys.argv) < 2 or sys.argv[1] not in ARMS:
    raise SystemExit("usage: python run_support_expansion_office_dslr.py <local|server|both> [seed]")
arm = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

args = make_args("office", seed)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)

feature_loader_list = [None] * args.num_users
for i, dom in enumerate(client_domains):
    if dom != "dslr":
        continue
    fixed = train_loader_list[i]
    expanded = resampled_loader(fixed, args, i)
    if arm == "local":
        feature_loader_list[i] = fixed
        train_loader_list[i] = expanded
    elif arm == "server":
        feature_loader_list[i] = expanded
    else:
        train_loader_list[i] = expanded

run(args, train_loader_list, test_loader_list, client_domains, feature_loader_list,
    lambda_G=0.05, tag="support_" + arm + "_M4_seed" + str(seed), out_dir="results")
