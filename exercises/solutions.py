"""Эталонные решения рабочей тетради. Не подглядывайте, пока не попробовали сами 🙂

Проверка решений: python tests/test_exercises.py
"""
import numpy as np


def sigmoid(z):                                                     # 1
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def softmax(z):                                                     # 2
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, y):                                       # 3
    z = logits - logits.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -log_p[np.arange(len(y)), y].mean()


def cosine_matrix(A, B):                                            # 4
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def pca(X, k):                                                      # 5
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, (S[:k] ** 2) / np.sum(S ** 2)


def linear_regression(X, y):                                        # 6
    return np.linalg.solve(X.T @ X, X.T @ y)


def gradient_descent(grad, x0, lr, steps):                          # 7
    x = np.array(x0, float)
    for _ in range(steps):
        x = x - lr * grad(x)
    return x


def numerical_gradient(f, x, h=1e-5):                               # 8
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float); e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):  # 9
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return w, m, v


def bayes_posterior(prior, sensitivity, false_positive_rate):        # 10
    num = sensitivity * prior
    return num / (num + false_positive_rate * (1 - prior))


def bootstrap_ci(x, stat, B=2000, alpha=0.05, seed=0):              # 11
    rng = np.random.default_rng(seed)
    vals = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(B)]
    return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))


def entropy_bits(p):                                                # 12
    p = np.asarray(p, float); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl_divergence(p, q):                                            # 13
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def precision_recall_f1(y, yhat):                                   # 14
    tp = np.sum((yhat == 1) & (y == 1)); fp = np.sum((yhat == 1) & (y == 0)); fn = np.sum((yhat == 0) & (y == 1))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def roc_auc(y, s):                                                  # 15
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def kmeans_step(X, C):                                              # 16
    lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
    C2 = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])
    return C2, lab


def attention(Q, K, V, causal=False):                               # 17
    s = Q @ K.T / np.sqrt(Q.shape[-1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape, bool)), s, -np.inf)
    return softmax(s) @ V


def conv_output_size(H, K, S=1, P=0, dilation=1):                   # 18
    return (H + 2 * P - dilation * (K - 1) - 1) // S + 1


def ndcg_at_k(rel, k):                                              # 19
    rel = np.asarray(rel, float)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[:k] - 1) * disc[:len(rel[:k])])
    ideal = np.sort(rel)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) * disc[:len(ideal)])
    return float(dcg / idcg) if idcg > 0 else 0.0


def conformal_quantile(residuals, alpha=0.1):                       # 20
    r = np.sort(np.asarray(residuals, float)); n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[min(k, n) - 1])
