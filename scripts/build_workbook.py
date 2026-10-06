#!/usr/bin/env python3
"""Собирает рабочую тетрадь exercises/workbook.ipynb: условие → заготовка функции → автопроверка.

Запуск:  python scripts/build_workbook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "justxor/math-for-ai-2026"
M = "https://github.com/" + REPO + "/blob/main/README.md#"

# (номер, заголовок, условие, модуль, заготовка, проверка)
EX = [
(1, "Устойчивая сигмоида", "Реализуйте $\\sigma(z) = 1/(1 + e^{-z})$ для массива так, чтобы не было переполнения при $z = \\pm 1000$.", "m13",
 "def sigmoid(z):\n    z = np.asarray(z, float)\n    # TODO\n    raise NotImplementedError", "check_sigmoid(sigmoid)"),
(2, "Softmax по строкам", "Матрица логитов $(N, K)$ → вероятности, каждая строка суммируется в 1. Не должно ломаться на логитах порядка 1000.", "m01",
 "def softmax(z):\n    # TODO: подсказка — keepdims=True\n    raise NotImplementedError", "check_softmax(softmax)"),
(3, "Кросс-энтропия по логитам", "Средний CE-лосс по батчу: `logits` $(N, K)$, `y` — метки классов. Считайте через log-softmax.", "m09",
 "def cross_entropy(logits, y):\n    # TODO\n    raise NotImplementedError", "check_cross_entropy(cross_entropy)"),
(4, "Матрица косинусных сходств", "Для $A$ $(n, d)$ и $B$ $(m, d)$ верните матрицу $(n, m)$ косинусов между всеми парами строк — без циклов.", "m02",
 "def cosine_matrix(A, B):\n    # TODO\n    raise NotImplementedError", "check_cosine_matrix(cosine_matrix)"),
(5, "PCA через SVD", "Верните проекцию на первые $k$ главных компонент и доли объяснённой дисперсии этих компонент.", "m02",
 "def pca(X, k):\n    # TODO: не забудьте центрировать\n    Z, ratio = None, None\n    raise NotImplementedError", "check_pca(pca)"),
(6, "Нормальное уравнение", "Найдите $\\mathbf{w} = (X^\\top X)^{-1}X^\\top\\mathbf{y}$. Используйте `np.linalg.solve`, а не `inv`.", "m10",
 "def linear_regression(X, y):\n    # TODO\n    raise NotImplementedError", "check_linear_regression(linear_regression)"),
(7, "Градиентный спуск", "Функция принимает градиент `grad(x)`, старт `x0`, шаг `lr`, число шагов и возвращает конечную точку.", "m05",
 "def gradient_descent(grad, x0, lr, steps):\n    x = np.array(x0, float)\n    # TODO\n    raise NotImplementedError", "check_gradient_descent(gradient_descent)"),
(8, "Численный градиент", "Центральная разность по каждой координате: $(f(x + h e_i) - f(x - h e_i))/2h$.", "m03",
 "def numerical_gradient(f, x, h=1e-5):\n    # TODO\n    raise NotImplementedError", "check_numerical_gradient(numerical_gradient)"),
(9, "Шаг Adam", "Один шаг Adam с коррекцией смещения. Верните новые `w, m, v`.", "m05",
 "def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):\n    # TODO\n    raise NotImplementedError", "check_adam_step(adam_step)"),
(10, "Формула Байеса", "Вероятность болезни при положительном тесте по распространённости, чувствительности и доле ложных срабатываний.", "m06",
 "def bayes_posterior(prior, sensitivity, false_positive_rate):\n    # TODO\n    raise NotImplementedError", "check_bayes(bayes_posterior)"),
(11, "Бутстрап-интервал", "95%-интервал статистики `stat` (например, `np.mean`) по `B` пересэмплированиям с возвращением. Верните `(low, high)`.", "m07",
 "def bootstrap_ci(x, stat, B=2000, alpha=0.05, seed=0):\n    rng = np.random.default_rng(seed)\n    # TODO\n    raise NotImplementedError", "check_bootstrap(bootstrap_ci)"),
(12, "Энтропия в битах", "$H = -\\sum p\\log_2 p$, причём $0 \\cdot \\log 0 = 0$.", "m08",
 "def entropy_bits(p):\n    # TODO\n    raise NotImplementedError", "check_entropy(entropy_bits)"),
(13, "KL-дивергенция", "$D_{KL}(P \\parallel Q) = \\sum p \\ln(p/q)$ в натах; члены с $p = 0$ пропускайте.", "m08",
 "def kl_divergence(p, q):\n    # TODO\n    raise NotImplementedError", "check_kl(kl_divergence)"),
(14, "Precision, recall, F1", "По бинарным `y` и `yhat` верните кортеж `(precision, recall, f1)`.", "m09",
 "def precision_recall_f1(y, yhat):\n    # TODO\n    raise NotImplementedError", "check_prf(precision_recall_f1)"),
(15, "ROC-AUC по определению", "Доля пар (позитив, негатив), где скор позитива выше; ничьи — с весом 1/2.", "m09",
 "def roc_auc(y, s):\n    # TODO\n    raise NotImplementedError", "check_auc(roc_auc)"),
(16, "Шаг k-means", "Назначьте точки ближайшим центрам и пересчитайте центры. Верните `(новые_центры, метки)`.", "m10",
 "def kmeans_step(X, C):\n    # TODO\n    raise NotImplementedError", "check_kmeans_step(kmeans_step)"),
(17, "Attention с маской", "$\\operatorname{softmax}(QK^\\top/\\sqrt{d})V$; при `causal=True` токен не видит будущие. Можно использовать свой `softmax` из упражнения 2.", "m11",
 "def attention(Q, K, V, causal=False):\n    # TODO\n    raise NotImplementedError", "check_attention(attention)"),
(18, "Размер выхода свёртки", "$\\lfloor (H + 2P - D(K-1) - 1)/S \\rfloor + 1$.", "m11",
 "def conv_output_size(H, K, S=1, P=0, dilation=1):\n    # TODO\n    raise NotImplementedError", "check_conv(conv_output_size)"),
(19, "NDCG@k", "Релевантности выдачи по порядку → NDCG@k с выигрышем $2^{rel} - 1$ и дисконтом $1/\\log_2(i + 1)$.", "m21",
 "def ndcg_at_k(rel, k):\n    # TODO\n    raise NotImplementedError", "check_ndcg(ndcg_at_k)"),
(20, "Квантиль для conformal prediction", "По ошибкам на калибровке верните $\\lceil (n+1)(1-\\alpha)\\rceil$-е значение по возрастанию.", "m22",
 "def conformal_quantile(residuals, alpha=0.1):\n    # TODO\n    raise NotImplementedError", "check_conformal(conformal_quantile)"),
]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.strip("\n").splitlines(True)}


def main():
    colab = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             f"(https://colab.research.google.com/github/{REPO}/blob/main/exercises/workbook.ipynb)")
    cells = [
        md(f"# ✍️ Рабочая тетрадь: 20 упражнений с автопроверкой\n\n{colab}\n\n"
           "Каждое упражнение: условие → заготовка функции (замените `raise NotImplementedError` своим кодом) → "
           "ячейка проверки. Зелёная галочка ✅ — всё верно; ❌ — подсказка, что исправить.\n\n"
           "Пишите на чистом NumPy. Эталонные решения — в [`solutions.py`](solutions.py), но сначала попробуйте сами."),
        code("# Загрузка автопроверок (в Colab файла рядом нет — скачиваем из репозитория)\n"
             "import os, urllib.request\n"
             "if not os.path.exists('checks.py'):\n"
             f"    urllib.request.urlretrieve('https://raw.githubusercontent.com/{REPO}/main/exercises/checks.py', 'checks.py')\n"
             "import numpy as np\nfrom checks import *"),
    ]
    for n, title, cond, mod, stub, check in EX:
        cells.append(md(f"## {n}. {title}\n\n{cond}\n\n📖 Теория: [модуль {mod[1:]}]({M}{mod})"))
        cells.append(code(stub))
        cells.append(code(check))
    cells.append(md("## 🎉 Готово\n\nЕсли все 20 проверок зелёные — переходите к [Практикуму на 100 задач]"
                    f"({M}practice) и [мини-проектам 81–90]({M}p-projects)."))
    for i, c in enumerate(cells):
        c["id"] = f"w{i:03d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "exercises" / "workbook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("упражнений:", len(EX))


if __name__ == "__main__":
    main()
