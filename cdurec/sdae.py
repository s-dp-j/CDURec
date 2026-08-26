"""Stacked Denoising AutoEncoder (SDAE) -- Section 3.3.2 of the paper.

A Denoising AutoEncoder (DAE, Vincent et al. 2008) learns a robust hidden
representation by reconstructing a clean input from a corrupted version of it.
An SDAE stacks several DAEs; training proceeds in three steps (Fig. 6):

1. *pretrain* -- greedily train one single-hidden-layer DAE at a time, feeding
   the hidden code of one DAE as the input of the next;
2. *unroll* -- the trained encoders are stacked to form a deep encoder;
3. (optionally) *finetune* -- back-propagation over the reconstruction loss.

Here we implement the greedy pretraining and expose the stacked encoder
weights (steps 1 and 2).  The final supervised fine-tuning is handled by
``DeepRegression``, which appends a linear regression head on top of the code
layer (Section 3.3.2, Fig. 7).
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class DAE:
    """A single denoising auto-encoder with tied weights and sigmoid units."""

    def __init__(self, n_in, n_hidden, corruption_level=0.2, lr=0.1,
                 n_epochs=60, batch_size=32, seed=0, verbose=False, desc="DAE"):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.corruption_level = corruption_level
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.desc = desc
        rng = np.random.RandomState(seed)

        bound = np.sqrt(6.0 / (n_in + n_hidden))
        self.W = rng.uniform(-bound, bound, size=(n_in, n_hidden))
        self.b = np.zeros(n_hidden)
        self.b_prime = np.zeros(n_in)

    def _corrupt(self, x, rng):
        mask = rng.binomial(1, 1.0 - self.corruption_level, size=x.shape)
        return x * mask

    def transform(self, X):
        """Encode (inference, no corruption)."""
        return _sigmoid(X @ self.W + self.b)

    def fit(self, X):
        """Greedy single-layer training with mini-batch SGD on reconstruction
        mean-squared-error."""
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.seed)
        n = X.shape[0]

        for e in range(self.n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = X[batch]
                m = xb.shape[0]

                x_tilde = self._corrupt(xb, rng)
                h = _sigmoid(x_tilde @ self.W + self.b)          # (m, n_hidden)
                z = _sigmoid(h @ self.W.T + self.b_prime)         # (m, n_in)

                # gradient of L = mean(0.5 * ||x - z||^2)
                d_a2 = (z - xb) / m * (z * (1.0 - z))             # (m, n_in)
                g_b_prime = d_a2.sum(axis=0)
                g_W_dec = d_a2.T @ h                              # (n_in, n_hidden)

                d_h = d_a2 @ self.W                               # (m, n_hidden)
                d_a1 = d_h * (h * (1.0 - h))                      # (m, n_hidden)
                g_b = d_a1.sum(axis=0)
                g_W_enc = x_tilde.T @ d_a1                        # (n_in, n_hidden)

                g_W = g_W_enc + g_W_dec

                self.W -= self.lr * g_W
                self.b -= self.lr * g_b
                self.b_prime -= self.lr * g_b_prime

            if self.verbose:
                h_full = _sigmoid(X @ self.W + self.b)
                z_full = _sigmoid(h_full @ self.W.T + self.b_prime)
                loss = float(np.mean((X - z_full) ** 2))
                print("\r[%s] epoch %d/%d  recon loss=%.4f"
                      % (self.desc, e + 1, self.n_epochs, loss), end="", flush=True)
        if self.verbose:
            print()

        return self


class SDAE:
    """Stacked denoising auto-encoder.

    ``layer_sizes`` lists the width of every layer from input to code, e.g.
    ``[16, 12, 8, 5]`` builds three DAE encoders 16->12, 12->8, 8->5.
    """

    def __init__(self, layer_sizes, corruption_level=0.2, lr=0.1,
                 n_epochs=60, batch_size=32, seed=0, verbose=False):
        self.layer_sizes = list(layer_sizes)
        self.corruption_level = corruption_level
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.daes = []

    def pretrain(self, X):
        """Greedy layer-wise pretraining (Fig. 6)."""
        X = np.asarray(X, dtype=float)
        inp = X
        for t in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[t]
            n_hidden = self.layer_sizes[t + 1]
            dae = DAE(n_in, n_hidden,
                      corruption_level=self.corruption_level,
                      lr=self.lr, n_epochs=self.n_epochs,
                      batch_size=self.batch_size, seed=self.seed + t,
                      verbose=self.verbose, desc="SDAE-L%d" % (t + 1))
            dae.fit(inp)
            self.daes.append(dae)
            inp = dae.transform(inp)
        return self

    def finetune(self, X, n_epochs=None, lr=None, batch_size=None):
        """Unroll the stacked encoders into a deep auto-encoder and fine-tune
        all weights end-to-end by back-propagation on the reconstruction MSE
        (Fig. 6, step 3 of the paper).  The decoders reuse the transposed
        encoder weights (tied weights), so the unrolled network is
        ``x -> h_1 -> ... -> code -> ... -> x_recon``.
        """
        X = np.asarray(X, dtype=float)
        n_epochs = self.n_epochs if n_epochs is None else n_epochs
        lr = self.lr if lr is None else lr
        batch_size = self.batch_size if batch_size is None else batch_size
        rng = np.random.RandomState(self.seed)
        n = X.shape[0]
        L = len(self.daes)
        Ws = [d.W for d in self.daes]
        bs = [d.b for d in self.daes]
        bps = [d.b_prime for d in self.daes]

        for e in range(n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                xb = X[batch]
                m = xb.shape[0]

                # forward: encode then decode with tied weights
                h = [xb]
                for t in range(L):
                    h.append(_sigmoid(h[-1] @ Ws[t] + bs[t]))
                r = [h[L]]
                for t in range(L - 1, -1, -1):
                    r.append(_sigmoid(r[-1] @ Ws[t].T + bps[t]))
                recon = r[L]

                gW = [np.zeros_like(W) for W in Ws]
                gb = [np.zeros_like(b) for b in bs]
                gbp = [np.zeros_like(bp) for bp in bps]

                # backward through the decode layers (t = 0 .. L-1)
                delta = (recon - xb) / m * (recon * (1.0 - recon))
                for t in range(L):
                    r_in = r[L - 1 - t]
                    r_out = r[L - t]
                    d_pre = delta * (r_out * (1.0 - r_out))
                    gbp[t] = d_pre.sum(axis=0)
                    gW[t] += d_pre.T @ r_in
                    delta = d_pre @ Ws[t]
                # backward through the encode layers (t = L-1 .. 0)
                for t in range(L - 1, -1, -1):
                    h_in = h[t]
                    h_out = h[t + 1]
                    d_pre = delta * (h_out * (1.0 - h_out))
                    gb[t] = d_pre.sum(axis=0)
                    gW[t] += h_in.T @ d_pre
                    delta = d_pre @ Ws[t].T

                for t in range(L):
                    Ws[t] -= lr * gW[t]
                    bs[t] -= lr * gb[t]
                    bps[t] -= lr * gbp[t]

            if self.verbose:
                hh = X
                for t in range(L):
                    hh = _sigmoid(hh @ Ws[t] + bs[t])
                rr = hh
                for t in range(L - 1, -1, -1):
                    rr = _sigmoid(rr @ Ws[t].T + bps[t])
                loss = float(np.mean((X - rr) ** 2))
                print("\r[SDAE finetune] epoch %d/%d  recon loss=%.4f"
                      % (e + 1, n_epochs, loss), end="", flush=True)
        if self.verbose:
            print()
        return self

    def encode(self, X):
        """Forward pass through the stacked encoders (no corruption)."""
        X = np.asarray(X, dtype=float)
        h = X
        for dae in self.daes:
            h = dae.transform(h)
        return h

    def encoder_weights(self):
        """Return the stacked encoder weights as a list of ``(W, b)`` pairs."""
        return [(dae.W.copy(), dae.b.copy()) for dae in self.daes]
