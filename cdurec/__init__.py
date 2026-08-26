"""CDURec: Cross-Domain and cross-User collaborative filtering
Recommendation for inactive users.

Reference:
    A cross-domain recommendation model for inactive users based on
    transferring the preference mapping relationship of active users.

This package contains the complete model described in the paper:

* ``funk_svd``          -- Funk-SVD (Eq. 2) and Restricted Funk-SVD (Algorithm 1)
* ``sdae``              -- stacked denoising auto-encoder (Eq. 4, Section 3.3.2)
* ``deep_regression``   -- deep regression mapping F (Eq. 5, Section 3.3.2)
* ``user_division``     -- active/inactive user division (Definitions 1/2, Sec. 3.1)
* ``cdurec``            -- the full CDURec pipeline (Algorithm 2)
"""

from .cdurec import CDURec
from .funk_svd import FunkSVD, RestrictedFunkSVD
from .sdae import SDAE, DAE
from .deep_regression import DeepRegression
from .user_division import divide_users

__all__ = [
    "CDURec",
    "FunkSVD",
    "RestrictedFunkSVD",
    "SDAE",
    "DAE",
    "DeepRegression",
    "divide_users",
]
