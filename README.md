


# CDURec: Cross-Domain and Cross-User Recommendation for Inactive Users

This repository provides the implementation of **CDURec**, a Cross-Domain and
cross-User collaborative filtering Recommendation framework designed to improve
recommendation performance for inactive users.

CDURec jointly exploits information from auxiliary domains and relatively rich
target-domain feedback from active users. Instead of treating all target-domain
users under the same transfer strategy, CDURec uses active users as supervised
anchors to learn cross-domain preference correspondence and subsequently transfers
the learned knowledge to inactive users.

---

## Overview

CDURec runs a five-step pipeline:

1. **User division** — split target-domain users by rating density
   `d_u = |I_u| / |I|` against a threshold `mu` into active (`d_u >= mu`) and
   inactive (`d_u < mu`) users (Definitions 1/2, Section 3.1).
2. **Active target-domain Funk-SVD** — factorize the active users' target
   sub-matrix into `p0_ua` and `q0a` (Eq. 2).
3-4. **Auxiliary-domain Funk-SVD** — factorize both auxiliary-domain matrices
   into `p1` / `p2` for all users (Eq. 2).
5. **Unsupervised SDAE pre-training** — stack the concatenated auxiliary
   factors `x = [p1; p2]` and learn a low-dimensional code (Eq. 4, Sec. 3.3.2).
6. **Supervised deep regression** — append a linear head to the SDAE encoder and
   learn the mapping `F : [p1; p2] -> p0` on the active users (Eq. 5).
7. **Transfer + Restricted Funk-SVD** — map inactive users' auxiliary factors
   through `F` to get `p0F`, then factorize the inactive users' target
   sub-matrix with `p0F` and `q0a` as soft constraints (Eqs. 6/7, Algorithm 1).

## Usage

```python
import numpy as np
from cdurec import CDURec

# R0, R1, R2: rating matrices of the target / auxiliary domains (nan = missing)
model = CDURec(
    activity_threshold=0.15,   # mu
    f_target=10,               # target-domain latent dimension k_target
    f_aux1=5, f_aux2=5,        # auxiliary-domain latent dimensions k_aux
    hidden_dims=(12, 8),       # SDAE hidden layers (from input to code)
    code_dim=5,                # SDAE code-layer width
    corruption_level=0.2,      # input corruption level rho
    lambda1=0.02,              # L2 regularizer of Restricted Funk-SVD
    lambda2=0.2,               # transfer-constraint coefficient
    seed=0,
)
model.fit(R0, R1, R2)

rating = model.predict(inactive_user, item)          # a single prediction
R_pred = model.predict_matrix()                      # full (n_inactive x n_items)
```

Missing ratings are represented by `numpy.nan` throughout.

## Notes on fidelity to the paper

- **Restricted Funk-SVD (Algorithm 1).**  The factors are warm-started at the
  transferred factor `pF` and the active-user item factor `qa`, updated one
  observed rating at a time in cyclic order with the decaying step size
  `gamma = 1/tau`, and stopped when the per-step change of the objective
  `|Delta J| = |J2 - J1|` falls below `tol = 1e-3` or the step counter `tau`
  exceeds `T_max = 10000`.  The paper writes `gamma = 1/tau`; the constructor
  parameter `lr0` is a constant scale factor (`gamma = lr0 / tau`), whose default
  `1.0` reproduces the paper literally.
- **Objective scaling.**  The objective (6) is a sum over the observed ratings
  without an explicit `1/2` factor; the constant factors are absorbed into the
  learning rate, as noted in the paper.
- **SDAE activation.**  The reference SDAE uses the sigmoid nonlinearity; the
  paper treats the activation `s(·)` as dataset-specific (sigmoid on Amazon,
  tanh on MovieLens).
---

## Requirements

The experiments in the paper were implemented using:

- Python
- PyTorch
- scikit-learn
