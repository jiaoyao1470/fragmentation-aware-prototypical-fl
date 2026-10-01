import sys
import os
sys.path.insert(0, os.getcwd())
from itertools import combinations

import torch
import torch.nn.functional as F

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients

if len(sys.argv) != 4 or sys.argv[3] not in ("1", "2", "4"):
    raise SystemExit("usage: python diag_dslr_geometry.py <label> <weights_dir> <1|2|4>")
label, weights_dir, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.argv = [sys.argv[0]]

args = args_parser()
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if torch.cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

train_loaders, client_domains, test_loaders = prepare_data_office_multi_clients(args, dslr_client_override=M)
dslr_test = test_loaders[list(dict.fromkeys(client_domains)).index("dslr")]
dslr_clients = [i for i, d in enumerate(client_domains) if d == "dslr"]


def class_features(model, loader):
    out = {}
    with torch.no_grad():
        for x, y in loader:
            rep, _ = model(x.to(args.device).float())
            for f, c in zip(F.normalize(rep, dim=1), y.tolist()):
                out.setdefault(c, []).append(f)
    return out


def cos_dist(a, b):
    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


rows = []
for ci in dslr_clients:
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(torch.load(os.path.join(weights_dir, f"best_local_model_client{ci}_dslr.pth"),
                                     map_location=args.device))
    model.eval()

    feats = class_features(model, train_loaders[ci])
    proto = {c: F.normalize(torch.stack(fs).mean(0), dim=0) for c, fs in feats.items()}
    wc = sum(sum(cos_dist(f, proto[c]) for f in fs) / len(fs) for c, fs in feats.items()) / len(feats)
    bc = sum(cos_dist(proto[a], proto[b]) for a, b in combinations(sorted(proto), 2)) / (len(proto) * (len(proto) - 1) / 2)

    classes = sorted(proto)
    bank = torch.stack([proto[c] for c in classes])
    head_ok = proto_ok = total = 0
    with torch.no_grad():
        for x, y in dslr_test:
            x, y = x.to(args.device).float(), y.to(args.device).long()
            rep, logits = model(x)
            pred_proto = torch.tensor([classes[i] for i in (F.normalize(rep, dim=1) @ bank.T).argmax(1).tolist()],
                                      device=args.device)
            head_ok += logits.argmax(1).eq(y).sum().item()
            proto_ok += pred_proto.eq(y).sum().item()
            total += y.size(0)
    rows.append((ci, head_ok / total, proto_ok / total, wc, bc))
    print(f"{label} dslr client {ci}: Acc_head={head_ok / total:.4f} Acc_proto={proto_ok / total:.4f} "
          f"WC={wc:.4f} BC={bc:.4f} BC/WC={bc / wc:.4f}")

n = len(rows)
print(f"{label} mean over {n} clients: Acc_head={sum(r[1] for r in rows) / n:.4f} "
      f"Acc_proto={sum(r[2] for r in rows) / n:.4f} WC={sum(r[3] for r in rows) / n:.4f} "
      f"BC={sum(r[4] for r in rows) / n:.4f}")
