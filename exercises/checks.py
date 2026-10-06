"""Автопроверки для рабочей тетради exercises/workbook.ipynb.

Каждая функция check_*(fn) запускает несколько тестов и печатает ✅,
либо падает с AssertionError и подсказкой, что пошло не так.
"""
import numpy as np

_rng = np.random.default_rng(42)


def _strict(fn, *args, hint="", **kw):
    """Вызов, при котором переполнение и NaN считаются ошибкой, а не предупреждением."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return fn(*args, **kw)
    except FloatingPointError as e:
        raise AssertionError(f"❌ численная неустойчивость ({e}). {hint}") from None


def _ok(name):
    print(f"✅ {name}: все тесты пройдены")


def _close(a, b, msg, atol=1e-6):
    assert np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol), f"❌ {msg}\n   получено: {a}\n   ожидалось: {b}"


# 1
def check_sigmoid(sigmoid):
    _close(sigmoid(np.array([0.0])), [0.5], "σ(0) должна быть 0.5")
    _close(_strict(sigmoid, np.array([-1000.0, 1000.0]), hint="разберите случаи z ≥ 0 и z < 0 (модуль 13)"), [0.0, 1.0],
           "σ(−1000) ≈ 0, σ(1000) ≈ 1")
    z = _rng.normal(size=10)
    _close(sigmoid(z) + sigmoid(-z), np.ones(10), "должно выполняться σ(z) + σ(−z) = 1")
    _ok("sigmoid")


# 2
def check_softmax(softmax):
    p = _strict(softmax, np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]), hint="вычтите максимум по строке")
    assert np.all(np.isfinite(p)), "❌ NaN/inf — вычтите максимум по строке"
    _close(p.sum(axis=1), [1, 1], "каждая строка должна суммироваться в 1 (axis=1, keepdims=True)")
    _close(p[0], [0.09003057, 0.24472847, 0.66524096], "неверные значения softmax")
    _close(p[0], p[1], "softmax не должен зависеть от сдвига логитов")
    _ok("softmax")


# 3
def check_cross_entropy(cross_entropy):
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]); y = np.array([1, 2])
    _close(cross_entropy(logits, y), (1.40760596 + np.log(3)) / 2, "средний CE по батчу посчитан неверно")
    big = np.array([[1000.0, 0.0]])
    _close(_strict(cross_entropy, big, np.array([1]), hint="считайте через log-softmax, а не log(softmax)"), 1000.0,
           "CE для логитов [1000, 0] и метки 1 равна 1000")
    _ok("cross_entropy")


# 4
def check_cosine_matrix(cosine_matrix):
    A = np.array([[1.0, 0.0], [1.0, 1.0]]); B = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0]])
    _close(cosine_matrix(A, B), [[1, 0, -0.70710678], [0.70710678, 0.70710678, -1]], "проверьте нормировку строк A и B")
    assert cosine_matrix(_rng.normal(size=(5, 8)), _rng.normal(size=(7, 8))).shape == (5, 7), "❌ форма должна быть (len(A), len(B))"
    _ok("cosine_matrix")


# 5
def check_pca(pca):
    X = _rng.normal(size=(300, 3)) @ np.array([[3.0, 0, 0], [1.0, 1.0, 0], [0, 0, 0.1]])
    Z, ratio = pca(X, 2)
    assert Z.shape == (300, 2), "❌ Z должна иметь форму (n, k)"
    _close(Z.mean(0), [0, 0], "данные нужно центрировать", atol=1e-8)
    assert ratio[0] >= ratio[1] and 0.99 < ratio.sum() <= 1.0 + 1e-9, "❌ доли дисперсии: по убыванию, сумма двух первых ≈ 1"
    _ok("pca")


# 6
def check_linear_regression(fit):
    X = np.c_[np.ones(100), _rng.normal(size=(100, 2))]; w = np.array([1.0, -2.0, 0.5])
    _close(fit(X, X @ w), w, "на данных без шума нормальное уравнение должно восстановить w точно")
    _ok("linear_regression")


# 7
def check_gradient_descent(gd):
    x = gd(lambda v: 2 * (v - 3), np.array([0.0]), lr=0.1, steps=200)
    _close(x, [3.0], "минимум (x − 3)² — в точке 3", atol=1e-4)
    x = gd(lambda v: np.array([v[0], 10 * v[1]]), np.array([5.0, 5.0]), lr=0.05, steps=500)
    _close(x, [0, 0], "двумерный случай: должно сойтись к нулю", atol=1e-3)
    _ok("gradient_descent")


# 8
def check_numerical_gradient(num_grad):
    f = lambda v: np.sum(v ** 3) + v[0] * v[1]
    x = np.array([1.0, 2.0, -1.0])
    _close(num_grad(f, x), [3 + 2, 12 + 1, 3], "используйте центральную разность (f(x+h) − f(x−h)) / 2h", atol=1e-5)
    _ok("numerical_gradient")


# 9
def check_adam_step(adam_step):
    w, m, v = np.zeros(2), np.zeros(2), np.zeros(2)
    w, m, v = adam_step(w, np.array([1.0, -1.0]), m, v, t=1, lr=0.1)
    _close(w, [-0.1, 0.1], "первый шаг Adam = −lr·sign(g) благодаря коррекции смещения", atol=1e-6)
    _close(m, [0.1, -0.1], "m = β₁m + (1 − β₁)g")
    _close(v, [0.001, 0.001], "v = β₂v + (1 − β₂)g²")
    _ok("adam_step")


# 10
def check_bayes(posterior):
    _close(posterior(prior=0.01, sensitivity=0.99, false_positive_rate=0.05), 0.0099 / (0.0099 + 0.0495), "формула Байеса")
    _close(posterior(0.5, 0.9, 0.1), 0.9, "симметричный случай")
    _ok("bayes_posterior")


# 11
def check_bootstrap(bootstrap_ci):
    x = _rng.normal(10, 2, 400)
    lo, hi = bootstrap_ci(x, np.mean, B=2000, alpha=0.05, seed=0)
    assert lo < 10 < hi or abs((lo + hi) / 2 - 10) < 0.3, "❌ интервал должен быть около истинного среднего"
    assert 0.25 < hi - lo < 0.5, f"❌ ширина 95%-интервала ≈ 2·1.96·σ/√n ≈ 0.39, получено {hi - lo:.3f}"
    _ok("bootstrap_ci")


# 12
def check_entropy(entropy_bits):
    _close(entropy_bits(np.array([0.5, 0.5])), 1.0, "энтропия честной монеты — 1 бит")
    _close(entropy_bits(np.array([1.0, 0.0])), 0.0, "0·log 0 считается равным 0")
    _close(entropy_bits(np.array([0.5, 0.25, 0.125, 0.125])), 1.75, "проверьте основание логарифма (биты)")
    _ok("entropy_bits")


# 13
def check_kl(kl):
    p, q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    _close(kl(p, q), 0.36808, "KL(P‖Q) = Σ p·ln(p/q)", atol=1e-4)
    _close(kl(q, p), 0.51083, "KL несимметрична: проверьте порядок аргументов", atol=1e-4)
    _close(kl(p, p), 0.0, "KL(P‖P) = 0")
    _ok("kl_divergence")


# 14
def check_prf(prf):
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1]); yhat = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    _close(prf(y, yhat), [2 / 3, 0.5, 4 / 7], "precision = TP/(TP+FP), recall = TP/(TP+FN), F1 — гармоническое среднее")
    _ok("precision_recall_f1")


# 15
def check_auc(roc_auc):
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    _close(roc_auc(y, s), 0.75, "AUC = доля пар (позитив, негатив), где скор позитива выше")
    _close(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5, "ничьи считаются как 1/2")
    _ok("roc_auc")


# 16
def check_kmeans_step(kmeans_step):
    X = np.array([[0.0, 0], [0, 1], [10, 10], [10, 11]]); C = np.array([[0.0, 0], [10, 10]])
    C2, lab = kmeans_step(X, C)
    assert list(lab) == [0, 0, 1, 1], "❌ назначьте каждую точку ближайшему центру"
    _close(C2, [[0, 0.5], [10, 10.5]], "новые центры — средние своих точек")
    _ok("kmeans_step")


# 17
def check_attention(attention):
    Q = np.array([[1.0, 0.0]]); K = np.array([[1.0, 0], [0, 1], [1, 1]]); V = np.array([[1.0, 2], [3, 4], [5, 6]])
    _close(attention(Q, K, V), [[3.0, 4.0]], "softmax(QKᵀ/√d)V — пример из задачи 36", atol=1e-3)
    T = 4; X = _rng.normal(size=(T, 3)); out = attention(X, X, X, causal=True)
    _close(out[0], X[0], "с causal-маской первый токен видит только себя")
    _ok("attention")


# 18
def check_conv(out_size):
    assert out_size(224, 7, 2, 3) == 112, "❌ ⌊(H + 2P − D(K−1) − 1)/S⌋ + 1"
    assert out_size(32, 3, 1, 1) == 32 and out_size(28, 5, 1, 0) == 24 and out_size(64, 3, 1, 2, dilation=2) == 64
    _ok("conv_output_size")


# 19
def check_ndcg(ndcg_at_k):
    _close(ndcg_at_k([3, 2, 0, 1, 0], 5), 0.99260, "DCG = Σ (2^rel − 1)/log₂(i + 1), делить на DCG идеального порядка", atol=1e-4)
    _close(ndcg_at_k([0, 0, 1], 3), 0.5, "единственный релевантный на 3-й позиции: 1/log₂4 = 0.5")
    _ok("ndcg_at_k")


# 20
def check_conformal(conformal_q):
    r = np.arange(1, 101, dtype=float)
    _close(conformal_q(r, alpha=0.1), 91.0, "возьмите ⌈(n+1)(1−α)⌉-е значение по возрастанию — здесь 91-е")
    _ok("conformal_quantile")
