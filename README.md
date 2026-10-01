# Client Fragmentation in Prototype-Based Federated Domain Generalisation

Patches and experiment scripts for studying client fragmentation in prototype-based federated domain generalisation, i.e. one domain split across several clients. The patches apply to a user-provided checkout of [FedPall](https://github.com/DistriAI/FedPall); the upstream code is not redistributed here beyond the short context lines every patch carries.

## Setup

```
git clone https://github.com/DistriAI/FedPall && cd FedPall
git checkout e08e675b26064498a558b4d00d77568bd5198fce
git apply --check --ignore-whitespace ../patches/*.patch
git apply --ignore-whitespace ../patches/*.patch
cp ../src/*.py exps/
```

Place Office-Caltech-10 / PACS under `data/` as FedPall expects. The patches were checked against the commit above only.

## Findings

1. Fragmenting domains over clients (see the split below) lowers accuracy: 65.88 -> 52.11 (Office-10), 59.96 -> 54.77 (PACS).
2. RA (domain-indexed discriminator, client-to-domain prototypes, cross-domain prototype G_k) recovers part of it: 55.13 / 55.62.
3. On DSLR (M=4), widening only the local-training support recovers essentially all of the oracle gap (mean 102% over 3 seeds); widening only server-side prototype support recovers about 16%.
4. StrongLocalAug (local view diversification, no extra data shared): Office-10 54.90 -> 61.32, PACS 55.41 -> 58.14. A baseline + StrongLocalAug control reaches 61.37 on Office-10, so the gain is not specific to RA.

Recovery in finding 3 is (arm - RA control) / (expanded support - RA control) on best cross-domain accuracy of DSLR, per seed, then averaged; it can exceed 100% when an arm outperforms the expanded-support oracle.

Numbers: robust standard (per-domain 5-round rolling mean), 3 seeds, test split used for checkpoint selection.

## Configurations

Flags are attributes of `args` (`domain_keyed_proto`, `no_discriminator_fix`); `lambda_G` is an argument of `ours()`. `common.make_args` defaults to the RA settings and `common.run` calls `ours()` with `preserve_feature_probe_state=True`. The ceiling, baseline, discriminator-fix-only and severity-sweep runs need no extra code: change the flags and the loaders, and pass `preserve_feature_probe_state=False` to `ours()` for the runs reported without it.

| Configuration | Data split | Settings |
|---|---|---|
| Ceiling | one client per domain | `domain_keyed_proto=False`, `no_discriminator_fix=True` |
| Baseline | fragmented | `domain_keyed_proto=False`, `no_discriminator_fix=True` |
| Discriminator fix only | fragmented | `domain_keyed_proto=False`, `no_discriminator_fix=False` |
| RA | fragmented | `domain_keyed_proto=True`, `no_discriminator_fix=False`, `lambda_G` = 0.05 (Office-10) / 0.0607 (PACS) |
| RA matched | fragmented | RA with `preserve_feature_probe_state=True`, no augmentation (PACS: `run_stronglocalaug_pacs.py baseline`) |
| RA + StrongLocalAug | fragmented | `run_stronglocalaug_office.py [seed]`, `run_stronglocalaug_pacs.py strongaug [seed]` |
| Baseline + StrongLocalAug | fragmented | the StrongLocalAug script with `domain_keyed_proto=False`, `no_discriminator_fix=True`, `lambda_G=0` (edit `common.make_args`) |
| Path A / Path B / expanded support | DSLR split into 4 clients | `run_support_expansion_office_dslr.py local`, `server`, `both` |
| FedDAP-inspired control | fragmented | `ours_feddap_inspired` in `federated_main.py`; `lamda_1=1`, `lamda_2=1`, `infoNCET=0.02`, attention temperature 0.01 |
| Severity sweep | DSLR split into M clients | `prepare_data_office_multi_clients(args, dslr_client_override=M)`, M = 1, 2, 4 |

Fragmented split: Office-10 has Caltech 3, Amazon 2, Webcam 1, DSLR 4 clients; PACS has Photo 3, Art Painting 2, Cartoon 1, Sketch 4.

## Scripts (`src/`)

- `common.py`: shared configuration, resampled / augmented loaders, run wrapper
- `run_support_expansion_office_dslr.py`: Path A / B / expanded-support runs
- `run_stronglocalaug_office.py`, `run_stronglocalaug_pacs.py`: StrongLocalAug
- `diag_dslr_geometry.py`: head vs nearest-prototype accuracy, within-class compactness, between-class separation (loads the best-round weights saved during training)

## Notes

See `NOTICE.md` for upstream attribution.
