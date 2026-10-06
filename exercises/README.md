# ✍️ Рабочая тетрадь с автопроверкой

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justxor/math-for-ai-2026/blob/main/exercises/workbook.ipynb)

20 упражнений по ключевым формулам курса. В каждом — условие, заготовка функции и ячейка проверки:

```text
## 2. Softmax по строкам          ← условие и ссылка на модуль
def softmax(z): ...               ← ваш код вместо raise NotImplementedError
check_softmax(softmax)            ← ✅ все тесты пройдены   или   ❌ подсказка
```

Проверки ловят не только неверные ответы, но и **численную неустойчивость**: наивный `softmax` без вычитания максимума не пройдёт.

| № | Упражнение | Модуль |
|---|-----------|--------|
| 1–3 | устойчивые sigmoid, softmax, кросс-энтропия | [01](../modules/01_notation.md#m01), [09](../modules/09_losses_metrics.md#m09), [13](../modules/13_numerics.md#m13) |
| 4–6 | косинусная матрица, PCA через SVD, нормальное уравнение | [02](../modules/02_linear_algebra.md#m02), [10](../modules/10_classic_ml.md#m10) |
| 7–9 | градиентный спуск, численный градиент, шаг Adam | [03](../modules/03_calculus.md#m03), [05](../modules/05_optimization.md#m05) |
| 10–11 | формула Байеса, бутстрап-интервал | [06](../modules/06_probability.md#m06), [07](../modules/07_statistics.md#m07) |
| 12–13 | энтропия, KL-дивергенция | [08](../modules/08_information_theory.md#m08) |
| 14–15 | precision/recall/F1, ROC-AUC | [09](../modules/09_losses_metrics.md#m09) |
| 16–18 | шаг k-means, attention с маской, размер свёртки | [10](../modules/10_classic_ml.md#m10), [11](../modules/11_deep_learning.md#m11) |
| 19–20 | NDCG@k, квантиль conformal prediction | [21](../modules/21_recommender_systems.md#m21), [22](../modules/22_bayes_uncertainty.md#m22) |

## Как работать

- **Colab:** кнопка выше — проверки скачаются автоматически.
- **Локально:** `jupyter lab exercises/workbook.ipynb` (файл `checks.py` лежит рядом).
- **Застряли?** Ссылка на модуль есть в каждом упражнении, эталон — в [`solutions.py`](solutions.py).

Для мейнтейнеров: тетрадь собирается `python scripts/build_workbook.py`, а `python tests/test_exercises.py` проверяет, что эталонные решения проходят все проверки (запускается в CI).
