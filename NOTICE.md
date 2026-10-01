# Notice and attribution

This repository contains patch files and experiment scripts written for this project. The patches are meant to be applied to a user-provided checkout of FedPall. Neither FedPall nor FedDAP files are included; each patch does carry a few lines of FedPall context (unchanged and removed lines) so that it can be applied.

## Upstream work

- **FedPall**, Zhang et al., ICCV 2025. https://github.com/DistriAI/FedPall. This work builds on the FedPall codebase. The upstream repository has no license file, so no FedPall source files are included beyond the patch context above.
- **FedDAP**, Le et al., CVPR 2026. https://github.com/quanghuy6997/FedDAP_CVPR2026. The `ours_feddap_inspired` path is a FedDAP-inspired comparison, not a reproduction. It follows FedDAP's domain-aware prototype aggregation and its two prototype losses, includes no FedDAP source code, and omits the parameter averaging. The upstream repository has no license file.
- **I2PFL**, Le et al., arXiv:2501.08521. The cross-domain prototype aggregation is a GPCL-style distance weighting adapted from it.

## Authorship

The patches and scripts in this repository were written for this project.
