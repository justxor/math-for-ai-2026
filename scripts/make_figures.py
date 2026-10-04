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


# ---------------------------------------------------------------------------
# Новые иллюстрации: база для начинающих и продвинутые модули
# ---------------------------------------------------------------------------

# 19. Галерея функций
def functions_gallery():
    x = np.linspace(-3, 3, 400)
    xp = np.linspace(0.01, 6, 400)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    axes[0].plot(x, 2 * x + 1, color=C[0], lw=2, label="y = 2x + 1")
    axes[0].plot(x, -0.5 * x, color=C[1], lw=2, label="y = −0.5x")
    axes[0].set_title("Линейные: наклон k, сдвиг b"); axes[0].legend(fontsize=9)
    for p, c in zip([1, 2, 3], C):
        axes[1].plot(x, x**p, color=c, lw=2, label=f"y = x^{p}")
    axes[1].set_ylim(-5, 9); axes[1].set_title("Степенные"); axes[1].legend(fontsize=9)
    axes[2].plot(x, np.exp(x), color=C[0], lw=2, label="eˣ")
    axes[2].plot(x, 2.0**x, color=C[2], lw=2, label="2ˣ")
    axes[2].plot(x, np.exp(-x), color=C[1], lw=2, ls="--", label="e⁻ˣ")
    axes[2].set_ylim(0, 10); axes[2].set_title("Экспоненты: рост и затухание"); axes[2].legend(fontsize=9)
    axes[3].plot(xp, np.log(xp), color=C[0], lw=2, label="ln x")
    axes[3].plot(xp, np.log2(xp), color=C[2], lw=2, label="log₂ x")
    axes[3].axhline(0, color="k", lw=0.6); axes[3].axvline(1, color="gray", ls=":")
    axes[3].text(1.1, -3.3, "ln 1 = 0", color="gray")
    axes[3].set_ylim(-4, 3); axes[3].set_title("Логарифмы: обратные к экспоненте"); axes[3].legend(fontsize=9)
    save(fig, "functions_gallery")


# 20. Производная = наклон касательной
def derivative_tangent():
    f = lambda x: 0.3 * x**3 - x + 1
    df = lambda x: 0.9 * x**2 - 1
    x = np.linspace(-2.5, 2.5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(x, f(x), color=C[0], lw=2.5, label="f(x)")
    x0 = 1.5
    for h, c in [(1.0, C[3]), (0.5, C[4]), (0.1, C[1])]:
        k = (f(x0 + h) - f(x0)) / h
        ax.plot(x, f(x0) + k * (x - x0), color=c, lw=1.2, ls="--", label=f"секущая h={h}: наклон {k:.2f}")
    ax.plot(x, f(x0) + df(x0) * (x - x0), color=C[2], lw=2, label=f"касательная: f′({x0}) = {df(x0):.2f}")
    ax.plot(x0, f(x0), "ko"); ax.set_ylim(-2, 4); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Производная — предел наклона секущей при h → 0")
    ax = axes[1]
    ax.plot(x, f(x), color=C[0], lw=2.5, label="f(x)")
    ax.plot(x, df(x), color=C[1], lw=2, label="f′(x)")
    for r in [-np.sqrt(1 / 0.9), np.sqrt(1 / 0.9)]:
        ax.axvline(r, color="gray", ls=":"); ax.plot(r, f(r), "o", color=C[2], ms=8)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-2, 4); ax.legend()
    ax.set_title("f′ = 0 в экстремумах; f′ > 0 — функция растёт, f′ < 0 — убывает")
    save(fig, "derivative_tangent")


# 21. Ряд Тейлора
def taylor():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, np.sin(x), color="k", lw=2.5, label="sin x")
    from math import factorial
    approx = np.zeros_like(x)
    for n, c in zip(range(0, 8), C * 2):
        k = 2 * n + 1
        approx = approx + (-1) ** n * x**k / factorial(k)
        if n in (0, 1, 2, 4):
            axes[0].plot(x, approx, color=c, lw=1.6, ls="--", label=f"до x^{k}")
    axes[0].set_ylim(-2, 2); axes[0].legend(fontsize=8); axes[0].set_title("Тейлор: чем больше членов, тем шире область точности")
    f = lambda x: np.log(1 + np.exp(x))
    x = np.linspace(-4, 4, 400); x0 = 1.0
    s = 1 / (1 + np.exp(-x0))
    axes[1].plot(x, f(x), color="k", lw=2.5, label="softplus(x)")
    axes[1].plot(x, f(x0) + s * (x - x0), color=C[0], ls="--", lw=1.8, label="1-й порядок (градиент)")
    axes[1].plot(x, f(x0) + s * (x - x0) + 0.5 * s * (1 - s) * (x - x0) ** 2, color=C[1], ls="--", lw=1.8, label="2-й порядок (гессиан)")
    axes[1].plot(x0, f(x0), "ko"); axes[1].set_ylim(-1, 5); axes[1].legend(fontsize=9)
    axes[1].set_title("GD использует 1-й порядок, метод Ньютона — 2-й")
    save(fig, "taylor")


# 22. Выпуклость и седловая точка
def convexity_saddle():
    fig = plt.figure(figsize=(15, 4.4))
    x = np.linspace(-2, 2, 300)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, x**2, color=C[0], lw=2.5, label="выпуклая x²")
    ax.plot(x, x**4 - 2 * x**2 + 0.3 * x, color=C[1], lw=2.5, label="невыпуклая: 2 минимума")
    ax.plot([-1.5, 1.2], [2.25, 1.44], "o--", color=C[0], lw=1)
    ax.set_ylim(-2, 4); ax.legend(fontsize=9); ax.set_title("Хорда над графиком ⇔ выпуклость")
    X, Y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
    for i, (Z, t) in enumerate([(X**2 + Y**2, "минимум: H ≻ 0"), (X**2 - Y**2, "седло: λ разных знаков")]):
        ax = fig.add_subplot(1, 3, i + 2, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.85, linewidth=0)
        ax.scatter([0], [0], [0], color="k", s=40)
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    save(fig, "convexity_saddle")


# 23. Forward vs reverse KL при подгонке гауссианы к бимодальному распределению
def kl_fit():
    x = np.linspace(-8, 8, 2000); dx = x[1] - x[0]
    P = 0.5 * stats.norm.pdf(x, -2.5, 0.8) + 0.5 * stats.norm.pdf(x, 2.5, 0.8)
    best_f, best_r = None, None
    for mu in np.linspace(-4, 4, 81):
        for s in np.linspace(0.3, 4, 75):
            Q = stats.norm.pdf(x, mu, s) + 1e-300
            f = np.sum(P * np.log((P + 1e-300) / Q)) * dx
            r = np.sum(Q * np.log(Q / (P + 1e-300))) * dx
            if best_f is None or f < best_f[0]: best_f = (f, mu, s)
            if best_r is None or r < best_r[0]: best_r = (r, mu, s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (val, mu, s), t, c in [(axes[0], best_f, "min KL(P‖Q) — forward: накрывает обе моды", C[0]),
                                   (axes[1], best_r, "min KL(Q‖P) — reverse: садится на одну моду", C[1])]:
        ax.fill_between(x, P, color="gray", alpha=0.3, label="P (данные)")
        ax.plot(x, stats.norm.pdf(x, mu, s), color=c, lw=2.5, label=f"Q = N({mu:.1f}, {s:.2f}²)")
        ax.set_title(t, fontsize=11); ax.legend(fontsize=9)
    save(fig, "kl_forward_reverse")


# 24. Калибровка
def calibration():
    rng = np.random.default_rng(0)
    n = 20000
    true_p = rng.beta(2, 2, n); y = rng.random(n) < true_p
    logit = np.log(true_p / (1 - true_p))
    over = 1 / (1 + np.exp(-2.5 * logit))      # переуверенная модель
    fig, ax = plt.subplots(figsize=(5.8, 5))
    bins = np.linspace(0, 1, 11)
    for p, c, lab in [(true_p, C[2], "откалибрована"), (over, C[1], "переуверена (как часто у глубоких сетей)")]:
        idx = np.digitize(p, bins) - 1
        conf = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
        acc = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 50]
        ece = sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(10) if (idx == b).sum() > 0)
        ax.plot(conf, acc, "o-", color=c, lw=2, label=f"{lab}, ECE={ece:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="идеал")
    ax.set_xlabel("предсказанная вероятность"); ax.set_ylabel("реальная доля позитивов")
    ax.set_title("Reliability diagram"); ax.legend(fontsize=8)
    save(fig, "calibration")


# 25. Ядровой трюк
def kernel_trick():
    rng = np.random.default_rng(2)
    n = 200
    r = np.r_[rng.uniform(0, 1, n), rng.uniform(1.6, 2.4, n)]
    t = rng.uniform(0, 2 * np.pi, 2 * n)
    X = np.c_[r * np.cos(t), r * np.sin(t)]; y = np.r_[np.zeros(n), np.ones(n)]
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(*X[y == 0].T, s=10, color=C[0]); ax.scatter(*X[y == 1].T, s=10, color=C[1])
    ax.set_aspect("equal"); ax.set_title("В исходном 2D прямой не разделить")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    z = (X**2).sum(1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0], s=8, color=C[0])
    ax.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1], s=8, color=C[1])
    G = np.linspace(-2.5, 2.5, 10); GX, GY = np.meshgrid(G, G)
    ax.plot_surface(GX, GY, np.full_like(GX, 1.7), alpha=0.25, color=C[2])
    ax.set_title("φ(x) = (x₁, x₂, x₁² + x₂²): разделяет плоскость")
    save(fig, "kernel_trick")


# 26. Гауссовский процесс
def gaussian_process():
    rng = np.random.default_rng(4)
    k = lambda a, b, l=0.8: np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / l**2)
    Xtr = np.array([-3.5, -2.0, -0.5, 1.0, 2.5]); ytr = np.sin(Xtr) + 0.05 * rng.normal(size=5)
    xs = np.linspace(-5, 5, 300)
    K = k(Xtr, Xtr) + 1e-3 * np.eye(5); Ks = k(xs, Xtr); Kss = k(xs, xs)
    mu = Ks @ np.linalg.solve(K, ytr); cov = Kss - Ks @ np.linalg.solve(K, Ks.T)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    prior = rng.multivariate_normal(np.zeros(300), Kss + 1e-6 * np.eye(300), 4)
    for p, c in zip(prior, C): axes[0].plot(xs, p, color=c, lw=1.5)
    axes[0].set_title("Априор: случайные гладкие функции из ядра RBF")
    axes[1].fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=C[0], alpha=0.2, label="±2σ")
    post = rng.multivariate_normal(mu, cov + 1e-6 * np.eye(300), 3)
    for p in post: axes[1].plot(xs, p, color=C[0], lw=0.8, alpha=0.6)
    axes[1].plot(xs, mu, color=C[0], lw=2.5, label="среднее"); axes[1].plot(xs, np.sin(xs), "k--", lw=1, label="истина")
    axes[1].scatter(Xtr, ytr, color=C[1], zorder=5, s=50, label="данные")
    axes[1].legend(fontsize=8, loc="lower left"); axes[1].set_title("Апостериор: неопределённость растёт вдали от данных")
    save(fig, "gaussian_process")


# 27. Value iteration на gridworld
def value_iteration():
    H, W, gamma = 5, 6, 0.9
    goal, pit, walls = (0, 5), (1, 5), {(1, 1), (2, 1), (3, 3)}
    V = np.zeros((H, W)); acts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def step(s, a):
        r, c = s[0] + a[0], s[1] + a[1]
        if not (0 <= r < H and 0 <= c < W) or (r, c) in walls: return s
        return (r, c)
    for _ in range(100):
        Vn = V.copy()
        for r in range(H):
            for c in range(W):
                s = (r, c)
                if s in walls or s in (goal, pit): continue
                Vn[r, c] = max(-0.04 + gamma * (1.0 if step(s, a) == goal else -1.0 if step(s, a) == pit else V[step(s, a)]) for a in acts)
        V = Vn
    fig, ax = plt.subplots(figsize=(7, 5.2))
    V[goal], V[pit] = 1.0, -1.0
    M = np.ma.array(V, mask=np.zeros_like(V, bool))
    for w in walls: M.mask[w] = True
    im = ax.imshow(M, cmap="RdYlGn", vmin=-1, vmax=1)
    arrows = {(-1, 0): "↑", (1, 0): "↓", (0, -1): "←", (0, 1): "→"}
    for r in range(H):
        for c in range(W):
            s = (r, c)
            if s in walls: ax.text(c, r, "■", ha="center", va="center", fontsize=20, color="gray"); continue
            if s == goal: ax.text(c, r, "+1", ha="center", va="center", fontsize=14, fontweight="bold"); continue
            if s == pit: ax.text(c, r, "−1", ha="center", va="center", fontsize=14, fontweight="bold"); continue
            best = max(acts, key=lambda a: (1.0 if step(s, a) == goal else -1.0 if step(s, a) == pit else V[step(s, a)]))
            ax.text(c, r - 0.15, arrows[best], ha="center", va="center", fontsize=16)
            ax.text(c, r + 0.25, f"{V[r, c]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("Value iteration: V(s) и жадная политика (γ = 0.9, штраф за шаг −0.04)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "value_iteration")


# 28. Спектральная кластеризация
def spectral_clustering():
    rng = np.random.default_rng(0)
    n = 150
    t = rng.uniform(0, np.pi, n)
    A = np.c_[np.cos(t), np.sin(t)] + 0.06 * rng.normal(size=(n, 2))
    B = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + 0.06 * rng.normal(size=(n, 2))
    X = np.r_[A, B]
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    Wm = np.exp(-D2 / (2 * 0.1**2)); np.fill_diagonal(Wm, 0)
    L = np.diag(Wm.sum(1)) - Wm
    Dm = np.diag(1 / np.sqrt(Wm.sum(1)))
    w, V = np.linalg.eigh(Dm @ L @ Dm)
    fied = V[:, 1]
    # k-means на исходных координатах для сравнения
    c = X[[0, n]]
    for _ in range(20):
        lab = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([X[lab == j].mean(0) for j in range(2)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(*X.T, c=lab, cmap="coolwarm", s=10); axes[0].set_title("k-means: режет «по прямой»")
    axes[1].scatter(*X.T, c=fied > 0, cmap="coolwarm", s=10); axes[1].set_title("Спектральная: знак вектора Фидлера")
    axes[2].plot(np.sort(fied), color=C[0], lw=2); axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_title(f"Отсортированный 2-й собственный вектор лапласиана\nλ₁={w[0]:.3f}, λ₂={w[1]:.3f}, λ₃={w[2]:.3f}")
    for ax in axes[:2]: ax.set_aspect("equal")
    save(fig, "spectral_clustering")


# 29. Double descent
def double_descent():
    rng = np.random.default_rng(1)
    n = 40
    f = lambda x: np.sin(2 * np.pi * x)
    xtr = rng.uniform(-1, 1, n); ytr = f(xtr) + 0.2 * rng.normal(size=n)
    xte = rng.uniform(-1, 1, 2000); yte = f(xte) + 0.2 * rng.normal(size=2000)
    widths = np.unique(np.r_[np.arange(4, 60, 4), np.geomspace(60, 3000, 14).astype(int)])
    te_err = []
    for p in widths:
        errs = []
        for seed in range(60):
            r = np.random.default_rng(seed)
            a = r.normal(size=p); b = r.uniform(-1, 1, p) * np.abs(a)
            phi = lambda x: np.maximum(0, np.outer(x, a) + b)        # случайные ReLU-признаки
            w = np.linalg.pinv(phi(xtr)) @ ytr                       # решение минимальной нормы
            errs.append(np.mean((phi(xte) @ w - yte) ** 2))
        te_err.append(np.median(errs))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(widths, te_err, "o-", color=C[1], ms=4, lw=1.8)
    ax.axvline(n, color="k", ls="--", lw=1)
    ax.text(n * 1.08, max(te_err) * 0.8, "порог интерполяции:\nпараметров ≈ N = 40", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("число случайных ReLU-признаков (лог. шкала)"); ax.set_ylabel("test MSE (медиана по 60 запускам)")
    ax.set_title("Double descent: пик у порога интерполяции, дальше ошибка снова падает")
    save(fig, "double_descent")


# 30. Wasserstein vs KL
def wasserstein():
    shifts = np.linspace(0, 6, 61)
    x = np.linspace(-10, 16, 4000); dx = x[1] - x[0]
    P = stats.norm.pdf(x, 0, 0.5)
    kl, js, w1 = [], [], []
    for s in shifts:
        Q = stats.norm.pdf(x, s, 0.5)
        kl.append(np.sum(P * np.log((P + 1e-300) / (Q + 1e-300))) * dx)
        M = 0.5 * (P + Q)
        js.append(0.5 * np.sum(P * np.log((P + 1e-300) / (M + 1e-300))) * dx + 0.5 * np.sum(Q * np.log((Q + 1e-300) / (M + 1e-300))) * dx)
        w1.append(s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].fill_between(x, P, alpha=0.4, color=C[0], label="P = N(0, 0.5²)")
    axes[0].fill_between(x, stats.norm.pdf(x, 4, 0.5), alpha=0.4, color=C[1], label="Q = N(θ, 0.5²), θ = 4")
    axes[0].annotate("", xy=(4, 0.5), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    axes[0].text(1.2, 0.56, "перевезти массу на θ"); axes[0].set_xlim(-3, 7); axes[0].legend(fontsize=9)
    axes[0].set_title("Оптимальный транспорт: «сколько работы» сдвинуть P в Q")
    axes[1].plot(shifts, w1, color=C[2], lw=2.5, label="W₁ = |θ| — гладко растёт")
    axes[1].plot(shifts, js, color=C[1], lw=2.5, label="JS → log 2: градиент ≈ 0")
    axes[1].plot(shifts, np.minimum(kl, 6), color=C[0], lw=1.5, ls="--", label="KL (обрезано): быстро растёт")
    axes[1].set_xlabel("сдвиг θ"); axes[1].legend(fontsize=9)
    axes[1].set_title("Почему WGAN: расстояние Вассерштейна даёт полезный градиент")
    save(fig, "wasserstein")


# ---------------------------------------------------------------------------
# Иллюстрации для BASICS.md
# ---------------------------------------------------------------------------

# 31. Единичная окружность и синус/косинус
def unit_circle():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="gray", lw=1.5)
    a = np.pi / 3
    ax.plot([0, np.cos(a)], [0, np.sin(a)], color="k", lw=2)
    ax.plot([np.cos(a), np.cos(a)], [0, np.sin(a)], color=C[1], lw=3, label=f"sin θ = {np.sin(a):.3f}")
    ax.plot([0, np.cos(a)], [0, 0], color=C[0], lw=3, label=f"cos θ = {np.cos(a):.3f}")
    ax.add_patch(plt.matplotlib.patches.Arc((0, 0), 0.5, 0.5, theta1=0, theta2=60, color=C[2], lw=2))
    ax.text(0.3, 0.1, "θ = 60° = π/3", color=C[2])
    ax.plot(np.cos(a), np.sin(a), "o", color="k")
    for ang, lab in [(0, "0"), (np.pi / 2, "π/2"), (np.pi, "π"), (3 * np.pi / 2, "3π/2")]:
        ax.text(1.18 * np.cos(ang), 1.18 * np.sin(ang), lab, ha="center", va="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Точка на окружности радиуса 1: (cos θ, sin θ)")
    ax = axes[1]
    x = np.linspace(0, 4 * np.pi, 500)
    ax.plot(x, np.sin(x), color=C[1], lw=2, label="sin x")
    ax.plot(x, np.cos(x), color=C[0], lw=2, label="cos x")
    ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi], ["0", "π", "2π", "3π", "4π"])
    ax.axhline(0, color="k", lw=0.6); ax.legend()
    ax.set_title("Период 2π: значения повторяются — так кодируют позицию токена")
    save(fig, "unit_circle")


# 32. Интеграл как площадь
def integral_area():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    f = lambda x: 0.5 * x**2 + 1
    x = np.linspace(0, 3, 300)
    for ax, n in zip(axes[:2], [6, 30]):
        ax.plot(x, f(x), color=C[0], lw=2.5)
        edges = np.linspace(0, 3, n + 1); w = edges[1] - edges[0]; mids = edges[:-1] + w / 2
        ax.bar(mids, f(mids), width=w, color=C[0], alpha=0.25, edgecolor=C[0])
        approx = (f(mids) * w).sum()
        ax.set_title(f"{n} прямоугольников: площадь ≈ {approx:.4f}\n(точно ∫₀³ = 7.5)")
    ax = axes[2]
    z = np.linspace(-4, 4, 400); pdf = stats.norm.pdf(z)
    ax.plot(z, pdf, color=C[1], lw=2.5)
    m = (z > -1) & (z < 1)
    ax.fill_between(z[m], pdf[m], color=C[1], alpha=0.3)
    ax.text(-0.75, 0.15, "P(−1 < X < 1)\n= 0.683", fontsize=10)
    ax.set_title("Вероятность = площадь под плотностью")
    save(fig, "integral_area")


# 33. Квадратное уравнение и системы уравнений
def equations():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.linspace(-3, 5, 300)
    for (a, b, c), col in zip([(1, -2, -3), (1, -2, 1), (1, -2, 3)], C):
        D = b**2 - 4 * a * c
        lab = f"x² {b:+}x {c:+}: D = {D}" + (" → 2 корня" if D > 0 else " → 1 корень" if D == 0 else " → нет корней")
        axes[0].plot(x, a * x**2 + b * x + c, color=col, lw=2, label=lab)
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_ylim(-5, 10); axes[0].legend(fontsize=8)
    axes[0].set_title("Корни = пересечения с осью x")
    x = np.linspace(-1, 5, 100)
    axes[1].plot(x, 4 - x, color=C[0], lw=2, label="x + y = 4")
    axes[1].plot(x, (x - 1) / 1, color=C[1], lw=2, label="x − y = 1")
    axes[1].plot(2.5, 1.5, "ko", ms=8); axes[1].text(2.65, 1.65, "(2.5; 1.5)")
    axes[1].legend(); axes[1].set_title("Система 2×2: одно решение — точка пересечения")
    axes[2].plot(x, 4 - x, color=C[0], lw=6, alpha=0.35, label="x + y = 4")
    axes[2].plot(x, 2 - x, color=C[1], lw=2, label="x + y = 2")
    axes[2].plot(x, 4 - x, color=C[2], lw=1, ls="--", label="2x + 2y = 8 (та же прямая)")
    axes[2].legend(fontsize=8); axes[2].set_title("Параллельные — нет решений;\nсовпадающие — бесконечно много (det = 0)")
    save(fig, "equations")


# 34. Прогрессии и пределы
def sequences_limits():
    n = np.arange(1, 31)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(n, 2 + 3 * (n - 1), "o-", color=C[0], ms=3, label="арифметическая: +3")
    axes[0].plot(n, 2 * 1.2 ** (n - 1), "o-", color=C[1], ms=3, label="геометрическая: ×1.2")
    axes[0].legend(); axes[0].set_title("Рост: линейный vs экспоненциальный")
    s = np.cumsum(0.5 ** (n - 1))
    axes[1].plot(n, s, "o-", color=C[2], ms=3); axes[1].axhline(2, color="k", ls="--", lw=1)
    axes[1].text(15, 1.85, "предел = 1/(1 − 0.5) = 2"); axes[1].set_title("1 + ½ + ¼ + … сходится к 2")
    nn = np.arange(1, 200)
    axes[2].plot(nn, (1 + 1 / nn) ** nn, color=C[3], lw=2); axes[2].axhline(np.e, color="k", ls="--", lw=1)
    axes[2].text(80, 2.55, "e ≈ 2.71828"); axes[2].set_title("(1 + 1/n)ⁿ → e: сложные проценты")
    save(fig, "sequences_limits")


# 35. Описательная статистика
def descriptive_stats():
    rng = np.random.default_rng(3)
    x = rng.lognormal(3.3, 0.5, 2000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ax = axes[0]
    ax.hist(x, bins=60, color=C[0], alpha=0.6)
    for v, lab, c in [(np.mean(x), "среднее", C[1]), (np.median(x), "медиана", C[2]), (np.percentile(x, 95), "p95", C[3])]:
        ax.axvline(v, color=c, lw=2, label=f"{lab} = {v:.1f}")
    ax.legend(fontsize=9); ax.set_title("Скошенное распределение (время ответа, мс)")
    ax = axes[1]
    ax.boxplot([rng.normal(0, 1, 300), rng.normal(0, 2, 300), np.r_[rng.normal(0, 1, 290), rng.normal(7, 1, 10)]])
    ax.set_xticks([1, 2, 3], ["σ = 1", "σ = 2", "с выбросами"]); ax.set_title("Ящик с усами: медиана, квартили, выбросы")
    ax = axes[2]
    for k, (rho, c) in enumerate(zip([0.9, 0.0, -0.7], C)):
        p = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 200)
        ax.scatter(p[:, 0] + 7 * k, p[:, 1], s=6, alpha=0.6, color=c)
        ax.text(7 * k, 3.6, f"ρ = {rho}", ha="center", color=c, fontweight="bold")
    ax.set_ylim(-4, 4.5); ax.set_xticks([]); ax.set_title("Корреляция: насколько точки «вытянуты» вдоль прямой")
    save(fig, "descriptive_stats")


# 36. Множества: диаграммы Венна
def venn():
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    titles = ["A ∪ B (объединение)", "A ∩ B (пересечение)", "A \\ B (разность)", "Aᶜ (дополнение)"]
    yy, xx = np.mgrid[-1.6:1.6:400j, -2.2:2.2:550j]
    inA = (xx + 0.6) ** 2 + yy**2 < 1; inB = (xx - 0.6) ** 2 + yy**2 < 1
    masks = [inA | inB, inA & inB, inA & ~inB, ~inA]
    for ax, t, m in zip(axes, titles, masks):
        ax.imshow(m, extent=[-2.2, 2.2, -1.6, 1.6], origin="lower", cmap="Blues", vmin=0, vmax=1.6, alpha=0.9)
        ax.add_patch(Circle((-0.6, 0), 1, fill=False, lw=2)); ax.add_patch(Circle((0.6, 0), 1, fill=False, lw=2))
        ax.text(-1.2, 1.1, "A", fontsize=13, fontweight="bold"); ax.text(1.05, 1.1, "B", fontsize=13, fontweight="bold")
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "venn")


if __name__ == "__main__":
    for fn in [dot_product, matrix_transform, eigen, pca, gd_contours, optimizers, activations,
               distributions, mvn, clt, entropy_kl, bias_variance, roc_pr, softmax_temp,
               attention, diffusion, lr_schedule, l1_l2,
               functions_gallery, derivative_tangent, taylor, convexity_saddle, kl_fit, calibration,
               kernel_trick, gaussian_process, value_iteration, spectral_clustering, double_descent,
               wasserstein, unit_circle, integral_area, equations, sequences_limits,
               descriptive_stats, venn]:
        fn()
