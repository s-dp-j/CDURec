"""Deep regression model (Section 3.3.2, Fig. 7).

A linear regression unit is appended after the code layer of a pretrained SDAE.
The network maps the concatenated auxiliary-domain latent factors
``[p1; p2]`` of an active user to the active user's target-domain latent factor
``p0``, and is fine-tuned with the supervised loss (5):

    (1 / (2 |Ua|)) * sum_{ua in Ua} || F(p1_ua, p2_ua) - p0_ua ||^2

The encoder weights are initialized from the pretrained SDAE (W'1, W'2, W'3)
while the linear regression weight (W'4) is initialized randomly.
"""

import numpy as np

from .sdae import _sigmoid


class DeepRegression:
    def __init__(self, encoder_weights, n_output, lr=0.05, n_epochs=200,
                 batch_size=32, seed=0, verbose=False, desc="DeepRegression"):
        # encoder layers: list of (W, b), from input to code
        self.encoder = [(W.copy(), b.copy()) for W, b in encoder_weights]
        self.n_output = n_output
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.desc = desc

        code_dim = self.encoder[-1][1].shape[0]
        rng = np.random.RandomState(seed)
        bound = np.sqrt(6.0 / (code_dim + n_output))
        self.W_out = rng.uniform(-bound, bound, size=(code_dim, n_output))
        self.b_out = np.zeros(n_output)

    def _forward(self, X):
        """Return the linear output and the list of layer activations
        (activations[0] is the input)."""
        acts = [X]
        h = X
        for W, b in self.encoder:
            h = _sigmoid(h @ W + b)
            acts.append(h)
        out = h @ self.W_out + self.b_out
        return out, acts

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out, _ = self._forward(X)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.RandomState(self.seed)
        n = X.shape[0]
        L = len(self.encoder)

        for e in range(self.n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = X[batch]
                yb = y[batch]
                m = xb.shape[0]

                out, acts = self._forward(xb)          # acts: [h0, h1, ..., hL]
                # loss = mean(0.5 * ||out - y||^2)
                d_out = (out - yb) / m                  # (m, n_output)

                g_W_out = acts[L].T @ d_out
                g_b_out = d_out.sum(axis=0)

                d_h = d_out @ self.W_out.T              # (m, code_dim)
                for t in range(L - 1, -1, -1):
                    h_t = acts[t + 1]                   # activation after layer t
                    d_a = d_h * (h_t * (1.0 - h_t))
                    g_W = acts[t].T @ d_a
                    g_b = d_a.sum(axis=0)
                    W, b = self.encoder[t]
                    W -= self.lr * g_W
                    b -= self.lr * g_b
                    d_h = d_a @ W.T                     # propagate to previous layer

                self.W_out -= self.lr * g_W_out
                self.b_out -= self.lr * g_b_out

            if self.verbose:
                mse = float(np.mean((self.predict(X) - y) ** 2))
                print("\r[%s] epoch %d/%d  mse=%.4f"
                      % (self.desc, e + 1, self.n_epochs, mse), end="", flush=True)
        if self.verbose:
            print()

        return self
