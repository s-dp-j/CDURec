"""Funk-SVD and the restricted Funk-SVD (Eq. 2 and Algorithm 1 of the paper).

Funk-SVD (Koren et al., 2009) factorizes a rating matrix ``R ~= P @ Q^T`` by
minimizing the squared reconstruction error plus an L2 regularizer (Eq. 2):

    min  sum_{(u,i) in D} (r_ui - p_u^T q_i)^2 + lambda (||p_u||^2 + ||q_i||^2)

with stochastic gradient descent.

The *restricted* variant (Algorithm 1) factorizes the *inactive* users' target
sub-matrix while pulling the final factors toward the transferred inactive-user
factor ``pF`` and the active-user item factor ``qa`` (optimization problem (6)).
Its step size ``gamma = 1/tau`` decays with the per-rating iteration counter
``tau``, and iteration stops when the per-step change of the objective
``|Delta J| = |J2 - J1|`` drops below ``tol`` or ``tau`` exceeds ``T_max``.
"""

import numpy as np


class FunkSVD:
    """Classic regularized matrix factorization solved by SGD (Eq. 2)."""

    def __init__(self, n_factors=8, n_epochs=30, lr=0.005, reg=0.02, seed=0,
                 verbose=False, desc="Funk-SVD"):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.desc = desc

    def fit(self, R):
        """Factorize rating matrix ``R`` (``nan`` = missing).

        Returns
        -------
        P : np.ndarray (n_users, n_factors)  -- user latent factors
        Q : np.ndarray (n_items, n_factors)  -- item latent factors
        """
        R = np.asarray(R, dtype=float)
        n_users, n_items = R.shape
        f = self.n_factors
        rng = np.random.RandomState(self.seed)

        self.user_factors_ = rng.normal(0.0, 0.1, size=(n_users, f))
        self.item_factors_ = rng.normal(0.0, 0.1, size=(n_items, f))

        triples = ratings_from_matrix(R)
        P, Q = self.user_factors_, self.item_factors_

        for e in range(self.n_epochs):
            rng.shuffle(triples)
            for u, i, r in triples:
                pu = P[u]
                qi = Q[i]
                err = r - float(pu @ qi)
                P[u] += self.lr * (err * qi - self.reg * pu)
                Q[i] += self.lr * (err * pu - self.reg * qi)
            if self.verbose:
                sse = sum((r - float(P[u] @ Q[i])) ** 2 for u, i, r in triples)
                rmse = (sse / len(triples)) ** 0.5
                print("\r[%s] epoch %d/%d  train rmse=%.4f"
                      % (self.desc, e + 1, self.n_epochs, rmse), end="", flush=True)
        if self.verbose:
            print()

        return self.user_factors_, self.item_factors_


class RestrictedFunkSVD:
    """Funk-SVD with transferred-factor and transferred-item constraints
    (Algorithm 1 in the paper).

    Solves optimization problem (6):

        min  sum_{D1} (r - p^T q)^2
             + lambda1 (||p||^2 + ||q||^2)
             + lambda2 (||p - pF||^2 + ||q - qa||^2)

    following Algorithm 1: the factors are warm-started at ``pF`` / ``qa``
    (Algorithm 1, line 1) and updated one observed rating at a time in cyclic
    order (line 4) with the decaying step size ``gamma = 1/tau`` (line 5; the
    counter ``tau`` increments on line 9).  Iteration stops when the per-step
    change of the objective ``|Delta J| = |J2 - J1|`` drops below ``tol`` or the
    step counter ``tau`` reaches ``max_iter`` (line 13).

    The paper writes ``gamma = 1/tau`` (initial ``gamma = 1``).  ``lr0`` is
    exposed as a constant scale factor (``gamma = lr0 / tau``) so the effective
    step size can be tuned; the default ``lr0 = 1.0`` reproduces the paper
    literally.
    """

    def __init__(self, n_factors=8, lr0=1.0, lambda1=0.02, lambda2=0.2,
                 tol=1e-3, max_iter=10000, seed=0, verbose=False,
                 desc="Restricted-Funk-SVD"):
        self.n_factors = n_factors
        self.lr0 = lr0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose
        self.desc = desc

    def fit(self, R, pF, qa):
        """Factorize ``R`` (inactive users x items, ``nan`` = missing).

        Parameters
        ----------
        R : np.ndarray (n_inactive, n_items)
        pF : np.ndarray (n_inactive, n_factors)  -- transferred user factors
        qa : np.ndarray (n_items, n_factors)     -- active-user item factors

        Returns
        -------
        P : np.ndarray (n_inactive, n_factors)
        Q : np.ndarray (n_items, n_factors)
        """
        R = np.asarray(R, dtype=float)
        pF = np.asarray(pF, dtype=float)
        qa = np.asarray(qa, dtype=float)
        l1, l2, lr0 = self.lambda1, self.lambda2, self.lr0

        # Warm start from the transferred knowledge (Algorithm 1, line 1).
        P = pF.copy()
        Q = qa.copy()

        triples = ratings_from_matrix(R)
        l = len(triples)
        if l == 0:
            return P, Q

        # Algorithm 1: tau=1, k=0 (line 1); cyclic rating order (line 4).
        k = 0
        for tau in range(1, self.max_iter + 1):
            k = (k + 1) % l
            u, i, r = triples[k]

            pu = P[u].copy()
            qi = Q[i].copy()
            err = r - float(pu @ qi)
            gamma = lr0 / tau

            # update rules (7) of the paper
            P[u] = pu + gamma * (err * qi - l1 * pu - l2 * (pu - pF[u]))
            Q[i] = qi + gamma * (err * pu - l1 * qi - l2 * (qi - qa[i]))

            # |Delta J| = |J2 - J1|, computed exactly (lines 10-11); only p_u
            # and q_i change, so the change equals the per-step delta below.
            delta = _objective_delta(pu, qi, P[u], Q[i], r, pF[u], qa[i], l1, l2)
            if self.verbose and tau % 1000 == 0:
                print("\r[%s] tau=%d  gamma=%.6f  |dJ|=%.6f"
                      % (self.desc, tau, gamma, abs(delta)), end="", flush=True)
            if abs(delta) < self.tol:
                break

        if self.verbose:
            print()
        return P, Q


def _objective_delta(pu, qi, pu_new, qi_new, r, pF_u, qa_i, l1, l2):
    """Exact per-step change of objective (6) when a single rating moves
    ``pu -> pu_new`` and ``qi -> qi_new`` (all other factors unchanged).

    Equivalent to ``|J2 - J1|`` in Algorithm 1 (lines 10-11) because only the
    ``(u, i)`` reconstruction term and the ``u``/``i`` regularization terms
    change.
    """
    d_se = (r - float(pu_new @ qi_new)) ** 2 - (r - float(pu @ qi)) ** 2
    d_l1 = l1 * (float(pu_new @ pu_new) - float(pu @ pu)
                 + float(qi_new @ qi_new) - float(qi @ qi))
    d_l2 = l2 * (float((pu_new - pF_u) @ (pu_new - pF_u))
                 - float((pu - pF_u) @ (pu - pF_u))
                 + float((qi_new - qa_i) @ (qi_new - qa_i))
                 - float((qi - qa_i) @ (qi - qa_i)))
    return d_se + d_l1 + d_l2


def ratings_from_matrix(R):
    """Convert a rating matrix (``nan`` = missing) to ``(u, i, r)`` triples."""
    R = np.asarray(R, dtype=float)
    u, i = np.nonzero(~np.isnan(R))
    return [(int(uu), int(ii), float(R[uu, ii])) for uu, ii in zip(u, i)]
