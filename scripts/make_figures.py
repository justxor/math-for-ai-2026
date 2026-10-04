#!/usr/bin/env python3
"""Генерирует все иллюстрации курса в каталог images/.

Запуск:  pip install numpy matplotlib scipy && python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(exist_ok=True)

C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓", name)


def arrow(ax, v, color, label=None, origin=(0, 0), **kw):
    ax.annotate("", xy=(origin[0] + v[0], origin[1] + v[1]), xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, **kw))
    if label:
        ax.text(origin[0] + v[0] * 1.08, origin[1] + v[1] * 1.08, label,
                color=color, fontsize=13, fontweight="bold")


# 1. Скалярное произведение и проекция
def dot_product():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    b = np.array([3.0, 0.6])
    for ax, (a, title) in zip(axes, [
        (np.array([2.0, 2.2]), "a·b > 0  (угол < 90°)"),
        (np.array([-0.4, 2.6]), "a·b ≈ 0  (ортогональны)"),
        (np.array([-2.2, 1.2]), "a·b < 0  (угол > 90°)")]):
        arrow(ax, b, C[0], "b")
        arrow(ax, a, C[1], "a")
        proj = (a @ b) / (b @ b) * b
        ax.plot([a[0], proj[0]], [a[1], proj[1]], "--", color="gray")
        arrow(ax, proj, C[2], None)
        cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        ax.set_title(f"{title}\ncos θ = {cos:.2f}")
        ax.set_xlim(-3, 4); ax.set_ylim(-1, 3.2); ax.set_aspect("equal")
    axes[0].text(0.3, -0.7, "зелёная — проекция a на b", color=C[2])
    save(fig, "dot_product")


# 2. Матрица как линейное преобразование
def matrix_transform():
    A = np.array([[1.5, 0.8], [0.3, 1.1]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)])
    grid = np.arange(-3, 4)
    for ax, M, title in [(axes[0], np.eye(2), "До: единичная сетка"),
                         (axes[1], A, "После: x ↦ Ax")]:
        for g in grid:
            line = M @ np.stack([np.full(2, g), [-3, 3]])
            ax.plot(*line, color="lightgray", lw=0.8)
            line = M @ np.stack([[-3, 3], np.full(2, g)])
            ax.plot(*line, color="lightgray", lw=0.8)
        ax.fill(*(M @ circle), color=C[0], alpha=0.15)
        ax.plot(*(M @ circle), color=C[0])
        arrow(ax, M @ [1, 0], C[1], "e₁" if M is not A else "Ae₁")
        arrow(ax, M @ [0, 1], C[2], "e₂" if M is not A else "Ae₂")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.set_title(title)
    axes[1].text(-2.9, -2.8, f"A = [[1.5, 0.8], [0.3, 1.1]],  det A = {np.linalg.det(A):.2f}\n"
                 "столбцы A — куда переходят e₁ и e₂", fontsize=9)
    save(fig, "matrix_transform")


# 3. Собственные векторы
def eigen():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    w, V = np.linalg.eigh(A)
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(0)
    for ang in np.linspace(0, np.pi, 13, endpoint=False):
        v = np.array([np.cos(ang), np.sin(ang)])
        Av = A @ v
        ax.annotate("", xy=Av, xytext=v, arrowprops=dict(arrowstyle="->", color="lightgray"))
        ax.plot(*v, "o", color="gray", ms=3)
    for i, c in zip(range(2), [C[1], C[2]]):
        arrow(ax, V[:, i], c, None)
        arrow(ax, A @ V[:, i], c, f"λ={w[i]:.0f}", alpha=0.6)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.set_title("Собственные векторы не меняют направление:\nAv = λv  (серые точки — повёрнуты, цветные — только растянуты)")
    save(fig, "eigenvectors")


# 4. PCA
def pca():
    rng = np.random.default_rng(1)
    X = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2.2]], 400)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (len(X) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.scatter(*Xc.T, s=8, alpha=0.4, color=C[0])
    for i, c in enumerate([C[1], C[2]]):
        arrow(ax, Vt[i] * 2 * np.sqrt(var[i]), c, f"PC{i+1}: {var[i]/var.sum():.0%}")
    ax.set_aspect("equal"); ax.set_title("Главные компоненты — оси наибольшей дисперсии")
    ax = axes[1]
    Z = Xc @ Vt[0]
    ax.scatter(Z, np.zeros_like(Z), s=8, alpha=0.3, color=C[1])
    ax.hist(Z, bins=40, color=C[1], alpha=0.5)
    ax.set_title("Проекция на PC1: 2D → 1D, сохранено ~%d%% дисперсии" % round(100 * var[0] / var.sum()))
    ax.set_yticks([])
    save(fig, "pca")


# 5. Градиентный спуск на контурах
def rosen_like(x, y):
    return 0.5 * x**2 + 4 * y**2


def gd_contours():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-2.5, 2.5, 200))
    Z = rosen_like(X, Y)
    for ax, lr, title in zip(axes, [0.05, 0.2, 0.24],
                             ["lr = 0.05: медленно ползёт по x", "lr = 0.2: быстро", "lr = 0.24: зигзаг поперёк долины\n(при lr > 0.25 — расходится)"]):
        ax.contour(X, Y, Z, levels=20, cmap="Blues", alpha=0.7)
        p = np.array([-4.5, 2.0]); path = [p.copy()]
        for _ in range(25):
            g = np.array([p[0], 8 * p[1]])
            p = p - lr * g
            path.append(p.copy())
            if np.abs(p).max() > 20:
                break
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color=C[1], ms=3, lw=1.2)
        ax.plot(0, 0, "*", color=C[2], ms=14)
        ax.set_xlim(-5, 5); ax.set_ylim(-2.5, 2.5); ax.set_title(title)
    fig.suptitle("Градиентный спуск θ ← θ − η∇L на вытянутой «долине» L = 0.5x² + 4y²", y=1.02)
    save(fig, "gradient_descent")


# 6. Сравнение оптимизаторов
def optimizers():
    def grad(p):
        return np.array([p[0], 8 * p[1]])

    def run(kind, lr, steps=60):
        p = np.array([-4.5, 2.0]); m = np.zeros(2); v = np.zeros(2); path = [p.copy()]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            g = grad(p)
            if kind == "SGD":
                p = p - lr * g
            elif kind == "Momentum":
                m = 0.9 * m + g; p = p - lr * m
            elif kind == "RMSProp":
                v = 0.9 * v + 0.1 * g**2; p = p - lr * g / (np.sqrt(v) + eps)
            elif kind == "Adam":
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g**2
                mh = m / (1 - b1**t); vh = v / (1 - b2**t)
                p = p - lr * mh / (np.sqrt(vh) + eps)
            path.append(p.copy())
        return np.array(path)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-2.5, 2.5, 200))
    ax.contour(X, Y, rosen_like(X, Y), levels=20, cmap="Greys", alpha=0.5)
    for (kind, lr), c in zip([("SGD", 0.05), ("Momentum", 0.006), ("RMSProp", 0.15), ("Adam", 0.3)], C):
        P = run(kind, lr)
        ax.plot(P[:, 0], P[:, 1], ".-", color=c, lw=1.5, ms=4, label=f"{kind} (lr={lr})")
    ax.plot(0, 0, "*", color="k", ms=14)
    ax.legend(loc="lower right"); ax.set_xlim(-5, 5); ax.set_ylim(-2.5, 2.5)
    ax.set_title("60 шагов: SGD ползёт по пологой оси, Momentum разгоняется (и проскакивает),\nRMSProp и Adam выравнивают масштаб шагов по осям")
    save(fig, "optimizers")


# 7. Функции активации и их производные
def activations():
    x = np.linspace(-5, 5, 400)
    sig = 1 / (1 + np.exp(-x))
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    silu = x * sig
    funcs = {
        "sigmoid": (sig, sig * (1 - sig)),
        "tanh": (np.tanh(x), 1 - np.tanh(x)**2),
        "ReLU": (np.maximum(0, x), (x > 0).astype(float)),
        "GELU": (gelu, np.gradient(gelu, x)),
        "SiLU / Swish": (silu, np.gradient(silu, x)),
    }
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for ax, (name, (f, d)) in zip(axes, funcs.items()):
        ax.plot(x, f, color=C[0], lw=2, label="f(x)")
        ax.plot(x, d, color=C[1], lw=1.6, ls="--", label="f′(x)")
        ax.set_title(name); ax.set_ylim(-1.5, 3)
    axes[0].legend(loc="upper left")
    fig.suptitle("Активации (сплошная) и их производные (пунктир): у sigmoid/tanh производная → 0 на краях — затухание градиента", y=1.04)
    save(fig, "activations")


# 8. Распределения
def distributions():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    k = np.arange(0, 15)
    axes[0].bar(k, stats.binom.pmf(k, 14, 0.3), color=C[0]); axes[0].set_title("Биномиальное B(n=14, p=0.3)")
    axes[1].bar(k, stats.poisson.pmf(k, 3), color=C[2]); axes[1].set_title("Пуассон (λ=3)")
    x = np.linspace(-5, 5, 400)
    for (mu, s), c in zip([(0, 1), (0, 2), (1.5, 0.6)], C):
        axes[2].plot(x, stats.norm.pdf(x, mu, s), color=c, lw=2, label=f"μ={mu}, σ={s}")
    axes[2].legend(fontsize=8); axes[2].set_title("Нормальное N(μ, σ²)")
    x = np.linspace(0, 1, 400)
    for (a, b), c in zip([(0.5, 0.5), (2, 5), (5, 2), (2, 2)], C):
        axes[3].plot(x, stats.beta.pdf(x, a, b), color=c, lw=2, label=f"α={a}, β={b}")
    axes[3].set_ylim(0, 3.5); axes[3].legend(fontsize=8); axes[3].set_title("Бета Beta(α, β) — «распределение над p»")
    save(fig, "distributions")


# 9. Многомерное нормальное
def mvn():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    pos = np.dstack([X, Y])
    for ax, (S, t) in zip(axes, [([[1, 0], [0, 1]], "Σ = I: независимые, «круг»"),
                                 ([[2, 0], [0, 0.4]], "диагональная Σ: разные дисперсии"),
                                 ([[1, 0.8], [0.8, 1]], "ρ = 0.8: корреляция наклоняет эллипс")]):
        ax.contourf(X, Y, stats.multivariate_normal([0, 0], S).pdf(pos), levels=12, cmap="Blues")
        ax.set_aspect("equal"); ax.set_title(t, fontsize=10)
    save(fig, "mvn")


# 10. ЦПТ
def clt():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, n in zip(axes, [1, 2, 5, 30]):
        means = rng.exponential(1.0, size=(20000, n)).mean(1)
        ax.hist(means, bins=60, density=True, color=C[0], alpha=0.6)
        x = np.linspace(means.min(), means.max(), 200)
        ax.plot(x, stats.norm.pdf(x, 1, 1 / np.sqrt(n)), color=C[1], lw=2)
        ax.set_title(f"среднее {n} экспоненциальных")
    fig.suptitle("ЦПТ: среднее независимых величин → нормальное N(μ, σ²/n), даже если исходное распределение кривое", y=1.05)
    save(fig, "clt")


# 11. Энтропия и KL
def entropy_kl():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    axes[0].plot(p, H, color=C[0], lw=2.5)
    axes[0].set_xlabel("p (вероятность «орла»)"); axes[0].set_ylabel("H, бит")
    axes[0].set_title("Энтропия монеты: максимум 1 бит при p = 0.5")
    x = np.linspace(-6, 8, 500)
    P = 0.6 * stats.norm.pdf(x, -1.5, 0.8) + 0.4 * stats.norm.pdf(x, 3, 1)
    Q = stats.norm.pdf(x, 0.3, 2.4)
    axes[1].fill_between(x, P, alpha=0.3, color=C[0], label="P (данные)")
    axes[1].plot(x, Q, color=C[1], lw=2, label="Q (модель)")
    dx = x[1] - x[0]
    kl_pq = np.sum(P * np.log((P + 1e-12) / (Q + 1e-12))) * dx
    kl_qp = np.sum(Q * np.log((Q + 1e-12) / (P + 1e-12))) * dx
    axes[1].set_title(f"KL несимметрична: KL(P‖Q) = {kl_pq:.2f},  KL(Q‖P) = {kl_qp:.2f}")
    axes[1].legend()
    save(fig, "entropy_kl")


# 12. Bias-variance / переобучение
def bias_variance():
    rng = np.random.default_rng(3)
    f = lambda x: np.sin(2 * np.pi * x)
    x = np.sort(rng.uniform(0, 1, 15)); y = f(x) + rng.normal(0, 0.25, 15)
    xs = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, d, t in zip(axes[:3], [1, 4, 14], ["степень 1: недообучение\n(высокий bias)",
                                                 "степень 4: в самый раз",
                                                 "степень 14: переобучение\n(высокая variance)"]):
        coef = np.polyfit(x, y, d)
        ax.plot(xs, f(xs), color="gray", ls="--", label="истина")
        ax.scatter(x, y, color=C[0], zorder=3)
        ax.plot(xs, np.polyval(coef, xs), color=C[1], lw=2)
        ax.set_ylim(-2, 2); ax.set_title(t, fontsize=10)
    xt = rng.uniform(0, 1, 500); yt = f(xt) + rng.normal(0, 0.25, 500)
    degs = range(0, 15); tr, te = [], []
    for d in degs:
        c = np.polyfit(x, y, d)
        tr.append(np.mean((np.polyval(c, x) - y)**2)); te.append(np.mean((np.polyval(c, xt) - yt)**2))
    axes[3].plot(degs, tr, "o-", color=C[0], label="train"); axes[3].plot(degs, te, "o-", color=C[1], label="test")
    axes[3].set_yscale("log"); axes[3].set_xlabel("сложность модели"); axes[3].legend()
    axes[3].set_title("Ошибка: train ↓ всегда, test — U-образная")
    save(fig, "bias_variance")


# 13. ROC и PR
def roc_pr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900), np.ones(100)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for sep, c in zip([0.5, 1.5, 3.0], C):
        s = np.r_[rng.normal(0, 1, 900), rng.normal(sep, 1, 100)]
        order = np.argsort(-s); ys = y[order]
        tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
        tpr = tp / ys.sum(); fpr = fp / (1 - ys).sum(); prec = tp / (tp + fp)
        auc = np.trapezoid(tpr, fpr)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"разделимость {sep}: AUC={auc:.2f}")
        axes[1].plot(tpr, prec, color=c, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="случайный: 0.5")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR (recall)"); axes[0].legend(fontsize=9)
    axes[0].set_title("ROC-кривая")
    axes[1].axhline(0.1, color="k", ls="--", lw=1); axes[1].text(0.6, 0.13, "базовый уровень = доля позитивов (10%)", fontsize=8)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("PR-кривая — честнее при дисбалансе")
    save(fig, "roc_pr")


# 14. Softmax и температура
def softmax_temp():
    z = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, T in zip(axes, [0.3, 1.0, 2.0, 10.0]):
        p = np.exp(z / T - (z / T).max()); p /= p.sum()
        ax.bar(range(5), p, color=C[0])
        H = -(p * np.log(p)).sum()
        ax.set_title(f"T = {T}  (H = {H:.2f})"); ax.set_xticks(range(5), ["A", "B", "C", "D", "E"])
    fig.suptitle("softmax(z / T): T → 0 — argmax (жадно), T → ∞ — равномерно (случайно). Логиты z = [2, 1, 0.5, 0.1, −1]", y=1.05)
    save(fig, "softmax_temperature")


# 15. Attention heatmap
def attention():
    rng = np.random.default_rng(7)
    tokens = ["Кот", "сел", "на", "ковёр", ",", "потому", "что", "он", "устал"]
    n, d = len(tokens), 16
    E = rng.normal(size=(n, d))
    E[7] = 0.85 * E[0] + 0.3 * rng.normal(size=d)  # «он» похож на «Кот»
    Wq = np.eye(d) + 0.1 * rng.normal(size=(d, d)); Wk = np.eye(d) + 0.1 * rng.normal(size=(d, d))
    S = (E @ Wq) @ (E @ Wk).T / np.sqrt(d)
    S = np.where(np.tril(np.ones((n, n))) == 1, S, -np.inf)
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(A, cmap="Blues")
    ax.set_xticks(range(n), tokens, rotation=45); ax.set_yticks(range(n), tokens)
    ax.set_xlabel("key (на кого смотрим)"); ax.set_ylabel("query (кто смотрит)")
    ax.set_title("Causal self-attention: softmax(QKᵀ/√d + маска)\nкаждая строка — распределение, сумма = 1")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "attention")


# 16. Диффузия: зашумление
def diffusion():
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)
    abar = np.cumprod(1 - betas)
    s = 0.008; tt = np.arange(T + 1) / T
    f = np.cos((tt + s) / (1 + s) * np.pi / 2) ** 2
    abar_cos = (f / f[0])[1:]
    yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
    x0 = ((xx**2 + yy**2 < 0.5) & (np.abs(xx) + np.abs(yy) > 0.45)).astype(float) * 2 - 1
    rng = np.random.default_rng(0); eps = rng.normal(size=x0.shape)
    fig = plt.figure(figsize=(15, 4.2))
    steps = [0, 50, 150, 300, 600, 999]
    for i, t in enumerate(steps):
        ax = fig.add_subplot(2, 6, i + 1)
        xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps
        ax.imshow(xt, cmap="gray"); ax.axis("off"); ax.set_title(f"t={t}", fontsize=9)
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(abar, color=C[0], lw=2, label="ᾱₜ линейное расписание β")
    ax.plot(abar_cos, color=C[1], lw=2, label="ᾱₜ косинусное расписание")
    ax.set_xlabel("шаг t"); ax.set_ylabel("доля сигнала ᾱₜ"); ax.legend()
    fig.suptitle("Диффузия: xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε. Модель учится предсказывать шум ε и идёт обратно", y=1.0)
    save(fig, "diffusion")


# 17. Learning rate schedule
def lr_schedule():
    steps = np.arange(10000); warm = 500; peak = 3e-4
    lr = np.where(steps < warm, peak * steps / warm,
                  1e-5 + 0.5 * (peak - 1e-5) * (1 + np.cos(np.pi * (steps - warm) / (10000 - warm))))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(steps, lr, color=C[0], lw=2)
    ax.axvspan(0, warm, color=C[3], alpha=0.15); ax.text(30, peak * 0.5, "warmup", color=C[3])
    ax.set_xlabel("шаг"); ax.set_ylabel("learning rate"); ax.set_title("Linear warmup + cosine decay — стандарт для трансформеров")
    save(fig, "lr_schedule")


# 18. Регуляризация L1 vs L2
def l1_l2():
    c0 = np.array([1.8, 0.6])
    H = np.array([[1.5, 0.4], [0.4, 1.0]])
    loss = lambda P: np.einsum("...i,ij,...j->...", P - c0, H, P - c0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    X, Y = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-2, 3, 300))
    L = loss(np.dstack([X, Y]))
    t = np.linspace(0, 2 * np.pi, 4000)
    for ax, kind, col in zip(axes, ["L1", "L2"], [C[1], C[2]]):
        if kind == "L1":
            r = 1 / (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        else:
            r = np.ones_like(t)
        B = np.stack([r * np.cos(t), r * np.sin(t)], 1)   # граница области ||w|| <= 1
        lb = loss(B); best = B[lb.argmin()]
        ax.fill(B[:, 0], B[:, 1], color=col, alpha=0.25)
        ax.contour(X, Y, L, levels=[lb.min() * k for k in (0.15, 0.4, 0.7)] + [lb.min(), lb.min() * 1.6],
                   colors=C[0], alpha=0.7)
        ax.plot(*best, "o", color=col, ms=10, zorder=5)
        ax.text(best[0] + 0.1, best[1] - 0.35, f"w* = ({best[0]:.2f}, {best[1]:.2f})", color=col)
        ax.set_title("L1 (Lasso): ромб — оптимум часто в углу\n→ часть весов ровно 0 (разреженность)" if kind == "L1"
                     else "L2 (Ridge): круг — оптимум на дуге\n→ веса маленькие, но не нулевые")
        ax.plot(*c0, "*", color="k", ms=12); ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-2, 3); ax.set_ylim(-2, 3)
    save(fig, "l1_l2")


if __name__ == "__main__":
    for fn in [dot_product, matrix_transform, eigen, pca, gd_contours, optimizers, activations,
               distributions, mvn, clt, entropy_kl, bias_variance, roc_pr, softmax_temp,
               attention, diffusion, lr_schedule, l1_l2]:
        fn()
