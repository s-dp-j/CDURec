"""CDURec -- Cross-Domain and cross-User recommendation for inactive users.

This is a direct implementation of Algorithm 2 in the paper.  The pipeline:

1. divide target-domain users into active / inactive (Sec. 3.1);
2. Funk-SVD on the active users' target sub-matrix -> ``p0_ua``, ``q0a_i``;
3-4. Funk-SVD on auxiliary domains 1 and 2 -> ``p1_u``, ``p2_u``;
5. build the unsupervised set ``TU`` by concatenating ``[p1_u; p2_u]``;
6. pretrain the SDAE on ``TU``;
7. build the supervised set ``TS`` from active users ``([p1_ua; p2_ua], p0_ua)``;
8. fine-tune the deep regression network (SDAE encoder + linear head) on ``TS``;
9. transfer the learned mapping to inactive users -> ``p0F_uina``;
10. restricted Funk-SVD on the inactive users' target sub-matrix, constrained by
    ``p0F_uina`` and ``q0a_i`` -> final ``p0_uina``, ``q0_i``.
"""

import numpy as np

from .funk_svd import FunkSVD, RestrictedFunkSVD
from .sdae import SDAE
from .deep_regression import DeepRegression
from .user_division import divide_users


class CDURec:
    def __init__(
        self,
        activity_threshold=0.25,
        f_target=8,
        f_aux1=8,
        f_aux2=8,
        hidden_dims=(12, 8),
        code_dim=5,
        corruption_level=0.2,
        svd_epochs=30,
        svd_lr=0.005,
        svd_reg=0.02,
        sdae_epochs=60,
        sdae_lr=0.1,
        reg_epochs=200,
        reg_lr=0.05,
        lambda1=0.02,
        lambda2=0.2,
        restricted_lr0=1.0,
        restricted_tol=1e-3,
        restricted_max_iter=10000,
        seed=0,
        verbose=False,
    ):
        self.activity_threshold = activity_threshold
        self.f_target = f_target
        self.f_aux1 = f_aux1
        self.f_aux2 = f_aux2
        self.hidden_dims = tuple(hidden_dims)
        self.code_dim = code_dim
        self.corruption_level = corruption_level
        self.svd_epochs = svd_epochs
        self.svd_lr = svd_lr
        self.svd_reg = svd_reg
        self.sdae_epochs = sdae_epochs
        self.sdae_lr = sdae_lr
        self.reg_epochs = reg_epochs
        self.reg_lr = reg_lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.restricted_lr0 = restricted_lr0
        self.restricted_tol = restricted_tol
        self.restricted_max_iter = restricted_max_iter
        self.seed = seed
        self.verbose = verbose

    def _funk(self, R, n_factors, desc):
        svd = FunkSVD(
            n_factors=n_factors,
            n_epochs=self.svd_epochs,
            lr=self.svd_lr,
            reg=self.svd_reg,
            seed=self.seed,
            verbose=self.verbose,
            desc=desc,
        )
        return svd.fit(R)

    def fit(self, R0, R1, R2):
        """Run the full CDURec pipeline (Algorithm 2).

        Parameters
        ----------
        R0, R1, R2 : rating matrices of target / aux1 / aux2 (``nan`` = missing).

        Returns
        -------
        self
        """
        R0 = np.asarray(R0, dtype=float)
        R1 = np.asarray(R1, dtype=float)
        R2 = np.asarray(R2, dtype=float)
        n_users = R0.shape[0]

        # 1. user division
        active, inactive, densities = divide_users(R0, self.activity_threshold)
        self.active_users_ = np.where(active)[0]
        self.inactive_users_ = np.where(inactive)[0]
        self.densities_ = densities
        self._inactive_row = {int(u): r for r, u in enumerate(self.inactive_users_)}

        # 2. Funk-SVD on active users' target sub-matrix
        R0a = R0[active]
        p0a, q0a = self._funk(R0a, self.f_target, "FunkSVD target-active")
        # p0a: (n_active, f0); q0a: (n_items, f0)

        # 3-4. Funk-SVD on the auxiliary domains (all users)
        p1, _ = self._funk(R1, self.f_aux1, "FunkSVD aux1")   # (n_users, f1)
        p2, _ = self._funk(R2, self.f_aux2, "FunkSVD aux2")   # (n_users, f2)

        # 5. unsupervised training set TU (all users)
        X_all = np.concatenate([p1, p2], axis=1)       # (n_users, f1 + f2)

        # 6. pretrain SDAE on TU
        layer_sizes = [X_all.shape[1]] + list(self.hidden_dims) + [self.code_dim]
        sdae = SDAE(
            layer_sizes,
            corruption_level=self.corruption_level,
            lr=self.sdae_lr,
            n_epochs=self.sdae_epochs,
            seed=self.seed,
            verbose=self.verbose,
        )
        sdae.pretrain(X_all)
        sdae.finetune(X_all)

        # 7. supervised training set TS (active users only)
        X_a = X_all[active]                            # (n_active, f1 + f2)
        y_a = p0a                                      # (n_active, f0)

        # 8. deep regression (SDAE encoder + linear head), fine-tuned on TS
        regression = DeepRegression(
            sdae.encoder_weights(),
            n_output=self.f_target,
            lr=self.reg_lr,
            n_epochs=self.reg_epochs,
            seed=self.seed,
            verbose=self.verbose,
            desc="DeepRegression",
        )
        regression.fit(X_a, y_a)

        # 9. transfer the mapping to inactive users
        X_ina = X_all[inactive]
        p0F = regression.predict(X_ina)                # (n_inactive, f0)

        # 10. restricted Funk-SVD on the inactive users' target sub-matrix
        R0ina = R0[inactive]
        rsvd = RestrictedFunkSVD(
            n_factors=self.f_target,
            lr0=self.restricted_lr0,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            tol=self.restricted_tol,
            max_iter=self.restricted_max_iter,
            seed=self.seed,
            verbose=self.verbose,
            desc="RestrictedFunkSVD",
        )
        p0_ina, q0 = rsvd.fit(R0ina, pF=p0F, qa=q0a)

        self.user_factors_ = p0_ina                     # (n_inactive, f0)
        self.item_factors_ = q0                         # (n_items, f0)
        self.p0F_ = p0F
        self.encoder_weights_ = sdae.encoder_weights()
        return self

    def predict(self, inactive_user, item):
        """Predicted rating of an (inactive) user ``u`` on item ``i``.

        ``inactive_user`` is the *global* target-domain user index.
        """
        row = self._inactive_row[int(inactive_user)]
        return float(self.user_factors_[row] @ self.item_factors_[int(item)])

    def predict_matrix(self):
        """Full (n_inactive x n_items) predicted rating matrix."""
        return self.user_factors_ @ self.item_factors_.T

    def predict_for(self, inactive_user, items):
        """Predicted ratings of one inactive user on a list of item indices."""
        row = self._inactive_row[int(inactive_user)]
        pu = self.user_factors_[row]
        return np.asarray([float(pu @ self.item_factors_[int(i)]) for i in items])
