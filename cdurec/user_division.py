"""User division into active and inactive users (Section 3.1 of the paper).

Definition 1 (rating density):  d_u = |I_u| / |I|
Definition 2 (active / inactive): a user u is *active* iff d_u >= mu,
    otherwise *inactive*. ``mu`` is the user activity threshold.
"""

import numpy as np


def divide_users(rating_matrix, threshold=0.25):
    """Split users of the *target* domain into active and inactive users.

    Parameters
    ----------
    rating_matrix : array-like of shape (n_users, n_items)
        Rating matrix of the target domain. Missing ratings are ``np.nan``.
    threshold : float
        The user activity threshold ``mu``.

    Returns
    -------
    active : np.ndarray (bool, shape (n_users,))
        True where the user is active.
    inactive : np.ndarray (bool, shape (n_users,))
        True where the user is inactive.
    densities : np.ndarray (float, shape (n_users,))
        The rating density d_u of every user.
    """
    R = np.asarray(rating_matrix, dtype=float)
    n_items = R.shape[1]
    rated_counts = np.sum(~np.isnan(R), axis=1)
    densities = rated_counts / float(n_items)

    active = densities >= threshold
    inactive = ~active
    return active, inactive, densities
