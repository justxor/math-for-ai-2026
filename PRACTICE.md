[← 23. 🧰 Прикладное: Фурье, свёртки и обработка сигналов](modules/23_fourier_signals.md) · [🏠 Оглавление](README.md#-оглавление) · [Шпаргалка на одной странице →](CHEATSHEET.md)

<a id="practice"></a>

# 🏋️ Практикум: 100 задач с решениями

> 📓 [Ноутбук с кодом всех решений](notebooks/practicum.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justxor/math-for-ai-2026/blob/main/notebooks/practicum.ipynb)

> Сначала решай на бумаге или в ноутбуке, потом открывай ▶️. Задачи — в формате реальных собеседований: часть «на доске», часть «напиши на NumPy за 10 минут».
> Все фрагменты кода начинаются с `import numpy as np` (опущено).

| Блок | Задачи |
|------|--------|
| [📐 Линейная алгебра](#p-la) | 1–6 |
| [📈 Анализ и backprop](#p-calc) | 7–12 |
| [⛰️ Оптимизация](#p-opt) | 13–17 |
| [🎲 Вероятности](#p-prob) | 18–23 |
| [📊 Статистика](#p-stat) | 24–28 |
| [🔤 Информация и лоссы](#p-info) | 29–32 |
| [🧠 Модели и Deep Learning](#p-dl) | 33–38 |
| [✨ LLM и генеративные](#p-gen) | 39–40 |
| [🟢 Простые задачи для разминки](#p-easy) | 41–50 |
| [🔴 Продвинутые задачи](#p-adv) | 51–60 |
| [🐞 Найди баг](#p-bugs) | 61–70 |
| [🧮 Оценка на салфетке](#p-fermi) | 71–80 |
| [🛠 Мини-проекты с нуля](#p-projects) | 81–90 |
| [🎤 Как на собеседовании](#p-interview) | 91–100 |

---

<a id="p-la"></a>

## 📐 Линейная алгебра

### Задача 1. Размерности
$X$: (32, 128), $W_1$: (128, 256), $W_2$: (256, 10). Какая форма у $XW_1W_2$, и в каком порядке умножать дешевле: $(XW_1)W_2$ или $X(W_1W_2)$?

<details><summary>▶️ Решение</summary>

Результат (32, 10). $(XW_1)W_2$: $32\cdot128\cdot256 + 32\cdot256\cdot10 = 1.13$M умножений. $X(W_1W_2)$: $128\cdot256\cdot10 + 32\cdot128\cdot10 = 0.37$M. Второй порядок в 3 раза дешевле — но только если $W_1W_2$ можно предвычислить (между ними нет нелинейности).
</details>

### Задача 2. Собственные числа на бумаге
Найдите собственные числа и векторы матрицы

```math
A = \begin{pmatrix}4 & 1\\ 2 & 3\end{pmatrix}
```

<details><summary>▶️ Решение</summary>

$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0 \Rightarrow \lambda_1 = 5,\ \lambda_2 = 2$. Проверка: след 7 = 5 + 2, определитель 10 = 5·2.
$\lambda = 5$: $(A - 5I)\mathbf{v} = 0 \Rightarrow -v_1 + v_2 = 0 \Rightarrow \mathbf{v} = (1, 1)$. $\lambda = 2$: $2v_1 + v_2 = 0 \Rightarrow \mathbf{v} = (1, -2)$.
</details>

### Задача 3. Степенной метод
Найдите наибольшее собственное число без `eig`.

<details><summary>▶️ Решение</summary>

```python
def power_iteration(A, iters=200):
    v = np.random.randn(A.shape[0])
    for _ in range(iters):
        v = A @ v; v /= np.linalg.norm(v)
    return v @ A @ v, v           # отношение Рэлея
A = np.array([[4., 1], [2, 3]])
print(power_iteration(A)[0])       # 5.0
```
Сходится со скоростью $\lvert\lambda_2/\lambda_1\rvert^k$. Так считают спектральную норму в spectral normalization для GAN (одна итерация на шаг обучения).
</details>

### Задача 4. Сжатие изображения через SVD
Сколько чисел нужно хранить для ранга $k$ у изображения 512×512 и при каком $k$ сжатие ещё выгодно?

<details><summary>▶️ Решение</summary>

$k(512 + 512 + 1) = 1025k$ против $262\,144$. Выгодно при $k < 255$. На практике $k = 50$ даёт сжатие в ~5 раз с хорошо узнаваемой картинкой.

```python
img = np.random.rand(512, 512)                    # подставьте реальное изображение в градациях серого
U, S, Vt = np.linalg.svd(img, full_matrices=False)
approx = lambda k: (U[:, :k] * S[:k]) @ Vt[:k]
energy = np.cumsum(S**2) / np.sum(S**2)           # доля «энергии» для выбора k
```
</details>

### Задача 5. Косинусная близость батчем
Даны эмбеддинги запросов $Q$ (100×384) и документов $D$ (10 000×384). Найдите топ-3 документа для каждого запроса без циклов.

<details><summary>▶️ Решение</summary>

```python
Q = np.random.randn(100, 384); D = np.random.randn(10_000, 384)
Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
S = Qn @ Dn.T                                      # (100, 10000)
top3 = np.argpartition(-S, 3, axis=1)[:, :3]       # O(n) вместо полной сортировки
top3 = np.take_along_axis(top3, np.argsort(-np.take_along_axis(S, top3, 1), 1), 1)
```
</details>

### Задача 6. Проекционная матрица
Постройте матрицу ортогональной проекции на пространство столбцов $X$ и проверьте её свойства.

<details><summary>▶️ Решение</summary>

$P = X(X^\top X)^{-1}X^\top$. Свойства: $P^2 = P$ (идемпотентна), $P^\top = P$, собственные числа 0 и 1, $\mathrm{tr}P = \mathrm{rank}X$.

```python
X = np.random.randn(10, 3)
P = X @ np.linalg.solve(X.T @ X, X.T)
print(np.allclose(P @ P, P), np.allclose(P, P.T), round(np.trace(P), 6))   # True True 3.0
```
$P\mathbf{y}$ — предсказания линейной регрессии; $\mathrm{tr}P$ — «число степеней свободы» модели.
</details>

---

<a id="p-calc"></a>

## 📈 Анализ и backprop

### Задача 7. Градиент на доске
$f(\mathbf{w}) = \log(1 + e^{-y\,\mathbf{w}^\top\mathbf{x}})$, $y \in \lbrace -1, +1 \rbrace$. Найдите $\nabla_\mathbf{w} f$.

<details><summary>▶️ Решение</summary>

```math
\nabla_\mathbf{w} f = \frac{-y\,\mathbf{x}\, e^{-y\mathbf{w}^\top\mathbf{x}}}{1 + e^{-y\mathbf{w}^\top\mathbf{x}}} = -y\,\mathbf{x}\,\sigma(-y\,\mathbf{w}^\top\mathbf{x})
```
Это логистическая регрессия в разметке ±1: правильно и уверенно классифицированные примеры ($y\mathbf{w}^\top\mathbf{x} \gg 0$) почти не дают градиента.
</details>

### Задача 8. Якобиан softmax
Выпишите якобиан softmax в матричной форме и проверьте численно.

<details><summary>▶️ Решение</summary>

$J = \mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. Симметричен, вырожден (строки суммируются в 0 — сдвиг всех логитов не меняет выход).

```python
z = np.random.randn(4); p = np.exp(z) / np.exp(z).sum()
J = np.diag(p) - np.outer(p, p)
h = 1e-6
J_num = np.array([(np.exp(z + h*e)/np.exp(z + h*e).sum() - np.exp(z - h*e)/np.exp(z - h*e).sum()) / (2*h)
                  for e in np.eye(4)]).T
print(np.allclose(J, J_num, atol=1e-8), np.allclose(J.sum(0), 0))
```
</details>

### Задача 9. Backward для MSE + линейный слой
$\hat{Y} = XW + \mathbf{b}$, $L = \frac{1}{N}\lVert \hat{Y} - Y\rVert_F^2$. Найдите $\partial L/\partial W$ и $\partial L/\partial \mathbf{b}$.

<details><summary>▶️ Решение</summary>

$G = \partial L/\partial\hat{Y} = \frac{2}{N}(\hat{Y} - Y)$. Тогда $\partial L/\partial W = X^\top G$, $\partial L/\partial\mathbf{b} = \sum_{\text{строки}} G$ (bias «раздаётся» на все строки — при обратном проходе суммируем).
</details>

### Задача 10. Мини-autograd
Напишите класс `Value` со сложением, умножением, `tanh` и `backward()` (в духе micrograd).

<details><summary>▶️ Решение</summary>

```python
import math
class Value:
    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad, self._parents, self._backward = data, 0.0, parents, backward
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def bw(): self.grad += (1 - t**2) * out.grad
        out._backward = bw; return out
    def backward(self):
        order, seen = [], set()
        def topo(v):
            if v not in seen:
                seen.add(v); [topo(p) for p in v._parents]; order.append(v)
        topo(self); self.grad = 1.0
        for v in reversed(order): v._backward()

x, w, b = Value(0.5), Value(-1.2), Value(0.3)
y = (x * w + b).tanh(); y.backward()
print(y.data, w.grad)   # w.grad = (1 - tanh²) · x
```
Ключевые моменты: топологическая сортировка, `+=` (переменная может использоваться несколько раз).
</details>

### Задача 11. Градиент L1 и субградиент
Чему равен градиент $\lambda\lVert\mathbf{w}\rVert_1$ и почему SGD с L1 не даёт точных нулей?

<details><summary>▶️ Решение</summary>

$\lambda\mathrm{sign}(\mathbf{w})$, в нуле — любое значение из $[-\lambda, \lambda]$ (субградиент). SGD «перепрыгивает» через 0 и колеблется около него. Точные нули даёт **проксимальный шаг** (soft thresholding): $w \leftarrow \mathrm{sign}(w)\max(\lvert w\rvert - \eta\lambda, 0)$ — так работают ISTA и координатный спуск в Lasso.
</details>

### Задача 12. Сколько памяти на активации
MLP: батч 64, 10 слоёв шириной 4096, FP32. Сколько памяти занимают сохранённые для backward активации?

<details><summary>▶️ Решение</summary>

На слой: $64 \cdot 4096 \cdot 4$ байта = 1 МБ (пред-активация; для ReLU нужна ещё маска или выход — ×2). 10 слоёв ≈ 10–20 МБ. Для трансформера на длинном контексте доминирует attention: $B \cdot h \cdot T^2$ на слой — при $T = 8192$ и 32 головах это ~4 ГБ на слой в FP16 для одного примера, отсюда FlashAttention и checkpointing.
</details>

---

<a id="p-opt"></a>

## ⛰️ Оптимизация

### Задача 13. Сходимость GD для квадратичной функции
$L(\mathbf{w}) = \frac{1}{2}\mathbf{w}^\top A\mathbf{w}$, собственные числа $A$ — 1 и 100. Найдите оптимальный постоянный LR и скорость сходимости.

<details><summary>▶️ Решение</summary>

По каждому собственному направлению множитель $\lvert 1 - \eta\lambda_i\rvert$. Оптимум уравнивает крайние: $1 - \eta\cdot1 = \eta\cdot100 - 1 \Rightarrow \eta = 2/101$. Множитель $\frac{\kappa - 1}{\kappa + 1} = 99/101 \approx 0.98$ — ~115 шагов на каждый порядок точности. Momentum улучшает до $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} = 9/11 \approx 0.82$ — ~12 шагов.
</details>

### Задача 14. SGD vs GD экспериментально
Сравните на логистической регрессии (10 000 примеров) полный GD и SGD с батчем 32 по числу **эпох** до лосса 0.3.

<details><summary>▶️ Решение</summary>

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(10_000, 20)); w_true = rng.normal(size=20)
y = (rng.random(10_000) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
sig = lambda z: 1 / (1 + np.exp(-z))
loss = lambda w: np.mean(np.logaddexp(0, X @ w) - y * (X @ w))

def train(bs, lr, epochs=20):
    w = np.zeros(20); hist = []
    for ep in range(epochs):
        idx = rng.permutation(len(y))
        for s in range(0, len(y), bs):
            b = idx[s:s + bs]
            w -= lr * X[b].T @ (sig(X[b] @ w) - y[b]) / len(b)
        hist.append(round(loss(w), 3))
    return hist
print("GD ", train(10_000, 1.0))
print("SGD", train(32, 0.1))
```
SGD опускается ниже 0.3 уже после первой эпохи, полному GD для этого нужно ~15 эпох: за эпоху SGD делает ~300 шагов вместо одного.
</details>

### Задача 15. Gradient clipping
Реализуйте clipping по глобальной норме и объясните, чем он лучше clipping по значению.

<details><summary>▶️ Решение</summary>

```python
def clip_by_global_norm(grads, max_norm):
    total = np.sqrt(sum((g**2).sum() for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads], total
```
По норме — сохраняет **направление** градиента, уменьшая только длину. По значению (`clip(g, -c, c)`) меняет направление. Стандарт для трансформеров: `max_norm = 1.0`, а сама норма — лучший ранний индикатор нестабильности.
</details>

### Задача 16. Weight decay как MAP
Покажите, что шаг SGD с L2-штрафом $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ — это «сжатие» весов перед шагом по градиенту.

<details><summary>▶️ Решение</summary>

$\mathbf{w} \leftarrow \mathbf{w} - \eta(\nabla L + \lambda\mathbf{w}) = (1 - \eta\lambda)\mathbf{w} - \eta\nabla L$. Для SGD L2 и weight decay совпадают; для Adam — нет (модуль 05, AdamW).
</details>

### Задача 17. Множители Лагранжа: максимум энтропии
Найдите распределение на $K$ исходах с максимальной энтропией.

<details><summary>▶️ Решение</summary>

$\mathcal{L} = -\sum p_i\log p_i + \lambda(\sum p_i - 1)$. $\partial/\partial p_i: -\log p_i - 1 + \lambda = 0 \Rightarrow p_i = e^{\lambda - 1}$ — одинаковы для всех $i$ ⇒ $p_i = 1/K$. С дополнительным ограничением на среднее $\sum p_i f_i = \mu$ получится $p_i \propto e^{\beta f_i}$ — **softmax** как распределение максимальной энтропии.
</details>

---

<a id="p-prob"></a>

## 🎲 Вероятности

### Задача 18. Парадокс дней рождения
Сколько людей нужно, чтобы вероятность совпадения дней рождения превысила 50%? При чём тут хэши и дедупликация датасетов?

<details><summary>▶️ Решение</summary>

$P(\text{нет совпадений}) = \prod_{i=0}^{n-1}(1 - i/365) \approx e^{-n^2/730}$ → $n = 23$. Для хэша с $N$ значениями коллизия ожидается уже при $\sim\sqrt{N}$ объектов: 32-битный хэш даёт коллизии на ~77 000 документах — для дедупликации веб-датасетов нужны 64+ бит (и MinHash для near-duplicates).
</details>

### Задача 19. Ожидаемое число уникальных
Из 1000 примеров сэмплируем 1000 с возвращением (бутстрап). Какая доля уникальных попадёт в выборку?

<details><summary>▶️ Решение</summary>

$P(\text{пример не выбран}) = (1 - 1/n)^n \to e^{-1} \approx 0.368$. В выборке ≈ 63.2% уникальных; оставшиеся 36.8% — «out-of-bag», на них Random Forest оценивает качество без отдельной валидации.
</details>

### Задача 20. Сэмплирование из softmax через Gumbel
Покажите экспериментально, что $\arg\max_i(z_i + g_i)$, $g_i = -\log(-\log u_i)$, распределён как $\mathrm{softmax}(\mathbf{z})$.

<details><summary>▶️ Решение</summary>

```python
z = np.array([1.0, 2.0, 0.5])
u = np.random.rand(200_000, 3)
samples = np.argmax(z - np.log(-np.log(u)), axis=1)
print(np.bincount(samples) / len(samples), np.exp(z) / np.exp(z).sum())   # совпадают
```
</details>

### Задача 21. Условное матожидание
Бросаем кубик, пока не выпадет 6. Сколько бросков в среднем? А пока не выпадут две шестёрки подряд?

<details><summary>▶️ Решение</summary>

Одна 6: геометрическое, $1/p = 6$. Две подряд: пусть $E$ — ожидание из начального состояния, $E_1$ — после одной 6. $E = 1 + \frac{1}{6}E_1 + \frac{5}{6}E$, $E_1 = 1 + \frac{5}{6}E$. Решая: $E = 42$. Тот же приём (уравнения на состояния) — основа уравнения Беллмана в RL.
</details>

### Задача 22. Ковариация суммы
$X, Y$ — стандартные нормальные с корреляцией $\rho$. Найдите $\mathrm{Var}(X + Y)$ и $\mathrm{Var}(X - Y)$.

<details><summary>▶️ Решение</summary>

$2 + 2\rho$ и $2 - 2\rho$. При $\rho = 1$ разность имеет нулевую дисперсию — на этом основаны **парные** тесты и CUPED: вычитание коррелированного компонента снижает шум.
</details>

### Задача 23. Репараметризация
Реализуйте сэмплирование из $\mathcal{N}(\boldsymbol\mu, \Sigma)$ через разложение Холецкого и проверьте ковариацию.

<details><summary>▶️ Решение</summary>

```python
mu = np.array([1.0, -2.0]); Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])
L = np.linalg.cholesky(Sigma)
x = mu + np.random.randn(100_000, 2) @ L.T
print(x.mean(0).round(2), np.cov(x.T).round(2))
```
</details>

---

<a id="p-stat"></a>

## 📊 Статистика

### Задача 24. MLE для экспоненциального
Время между запросами к API ~ Exp(λ). Наблюдения: 2, 3, 1, 4, 5 с. Найдите $\hat\lambda$.

<details><summary>▶️ Решение</summary>

$\ell(\lambda) = n\log\lambda - \lambda\sum x_i$, $\ell' = n/\lambda - \sum x_i = 0 \Rightarrow \hat\lambda = 1/\bar{x} = 1/3$ запроса в секунду.
</details>

### Задача 25. Доверительный интервал для accuracy
Модель дала 870 правильных ответов из 1000. 95%-ный интервал для accuracy? Сколько примеров нужно для точности ±1%?

<details><summary>▶️ Решение</summary>

$\hat p = 0.87$, $SE = \sqrt{0.87\cdot0.13/1000} = 0.0106$, интервал $0.87 \pm 0.021$ → [0.849, 0.891]. Для ±0.01: $n = 1.96^2\cdot0.87\cdot0.13/0.01^2 \approx 4345$. **Вывод для бенчмарков:** разница в 1–2 п.п. на тест-сете из 1000 примеров часто статистически незначима.
</details>

### Задача 26. Сравнение двух моделей на одном тест-сете
Модели A и B на 1000 примерах: A права, B ошиблась — 60 раз; B права, A ошиблась — 35 раз. Значима ли разница?

<details><summary>▶️ Решение</summary>

Тест Макнемара (используются только несогласованные пары): $\chi^2 = \frac{(\lvert 60 - 35\rvert - 1)^2}{60 + 35} = \frac{576}{95} \approx 6.06$, p ≈ 0.014 < 0.05 — A значимо лучше. Независимые тесты (z-тест пропорций) здесь некорректны — примеры общие.
</details>

### Задача 27. Множественные сравнения
Проверили 20 гипотез, p-values отсортированы: 0.001, 0.008, 0.012, 0.03, 0.04, … Какие значимы по Бонферрони и по Бенджамини–Хохбергу (FDR 5%)?

<details><summary>▶️ Решение</summary>

Бонферрони: порог $0.05/20 = 0.0025$ → только 0.001. BH: сравниваем $p_{(k)}$ с $\frac{k}{20}\cdot0.05$: 0.0025, 0.005, 0.0075, 0.01, 0.0125… → $p_{(1)} = 0.001 \le 0.0025$ ✓, $p_{(2)} = 0.008 > 0.005$ ✗, $p_{(3)} = 0.012 > 0.0075$ ✗, $p_{(4)} = 0.03 > 0.01$ ✗, $p_{(5)} = 0.04 > 0.0125$ ✗ (и остальные больше своих порогов) → тоже одна гипотеза. При менее «размазанных» p-values BH обычно находит больше открытий, контролируя долю ложных среди них, а не вероятность хотя бы одной ошибки.
</details>

### Задача 28. Регрессия к среднему
Отобрали 10% худших по метрике пользователей, провели «улучшение» — их метрика выросла. Значит ли это, что улучшение работает?

<details><summary>▶️ Решение</summary>

Нет: отбор по крайним значениям шумной метрики гарантирует, что при повторном измерении они в среднем станут ближе к среднему **без всякого воздействия**. Нужна контрольная группа из тех же «худших», рандомизированная.
</details>

---

<a id="p-info"></a>

## 🔤 Информация и лоссы

### Задача 29. CE и KL
Покажите, что $\nabla_\theta H(P, Q_\theta) = \nabla_\theta D_{\mathrm{KL}}(P\parallel Q_\theta)$.

<details><summary>▶️ Решение</summary>

$H(P, Q_\theta) = H(P) + D_{\mathrm{KL}}(P\parallel Q_\theta)$, а $H(P)$ от $\theta$ не зависит. Поэтому оптимизировать CE и KL(данные‖модель) — одно и то же, отличаются только значения лосса (CE ≥ H(P) > 0 даже у идеальной модели на шумных данных).
</details>

### Задача 30. Perplexity
Средний CE-лосс модели на валидации — 2.3 ната на токен. Какая perplexity? Как изменится PPL, если лосс снизится на 0.1?

<details><summary>▶️ Решение</summary>

$e^{2.3} \approx 9.97$. После снижения: $e^{2.2} \approx 9.03$ — на ~9.5% меньше ($e^{-0.1} \approx 0.905$). Снижение лосса на константу = **умножение** PPL на константу.
</details>

### Задача 31. Label smoothing
С ε = 0.1 и 10 классами какой минимальный CE-лосс достижим? При каких логитах?

<details><summary>▶️ Решение</summary>

Цель $\mathbf{q} = (0.91, 0.01, \dots, 0.01)$. Минимум CE — при $\mathbf{p} = \mathbf{q}$, равен энтропии $H(\mathbf{q}) = -0.91\ln0.91 - 9\cdot0.01\ln0.01 \approx 0.086 + 0.414 = 0.50$. Логиты: $z_{\text{true}} - z_{\text{other}} = \ln(0.91/0.01) \approx 4.5$ — **конечный** зазор, модель не тянет логиты к бесконечности.
</details>

### Задача 32. Focal loss
При $\gamma = 2$ во сколько раз focal loss уменьшает вклад примера с $p_t = 0.9$ по сравнению с $p_t = 0.5$?

<details><summary>▶️ Решение</summary>

Множитель $(1 - p_t)^2$: 0.01 против 0.25 — в 25 раз; плюс сам $-\log p_t$ меньше (0.105 против 0.693). Итог: вклад лёгкого примера в ~165 раз меньше. Модель фокусируется на трудных и редких.
</details>

---

<a id="p-dl"></a>

## 🧠 Модели и Deep Learning

### Задача 33. Логистическая регрессия с нуля
Обучите на NumPy с L2-регуляризацией и сравните со sklearn.

<details><summary>▶️ Решение</summary>

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X, y = make_classification(2000, 10, random_state=0)
w, b, lam, lr = np.zeros(10), 0.0, 1e-2, 0.5
for _ in range(3000):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    w -= lr * (X.T @ (p - y) / len(y) + lam * w)
    b -= lr * (p - y).mean()
sk = LogisticRegression(C=1 / (lam * len(y))).fit(X, y)   # C = 1/(λN) из-за разной нормировки
print(np.round(w[:4], 3), np.round(sk.coef_[0][:4], 3))   # близки
```
**Ловушка:** sklearn минимизирует $C\sum_i \ell_i + \frac{1}{2}\lVert\mathbf{w}\rVert^2$, а не среднее + $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ — отсюда $C = 1/(\lambda N)$.
</details>

### Задача 34. Выход свёрточной сети
Вход 3×224×224. Conv(64, k=7, s=2, p=3) → MaxPool(k=3, s=2, p=1) → Conv(128, k=3, s=2, p=1). Размер выхода и число параметров свёрток?

<details><summary>▶️ Решение</summary>

224 → $\lfloor(224 + 6 - 7)/2\rfloor + 1 = 112$ → $\lfloor(112 + 2 - 3)/2\rfloor + 1 = 56$ → $\lfloor(56 + 2 - 3)/2\rfloor + 1 = 28$. Выход 128×28×28. Параметры: $64\cdot(3\cdot49 + 1) = 9472$ и $128\cdot(64\cdot9 + 1) = 73\,856$. (Это начало ResNet.)
</details>

### Задача 35. Почему residual помогает — численно
Сравните норму градиента на первом слое у сети из 50 слоёв с residual-связями и без.

<details><summary>▶️ Решение</summary>

```python
import torch
def grad_norm(residual, depth=50, d=64):
    torch.manual_seed(0)
    layers = [torch.nn.Linear(d, d) for _ in range(depth)]
    x = torch.randn(32, d, requires_grad=True); h = x
    for l in layers:
        h = h + torch.tanh(l(h)) * 0.1 if residual else torch.tanh(l(h))
    h.sum().backward()
    return x.grad.norm().item()
print(grad_norm(False), grad_norm(True))   # без residual — на порядки меньше
```
</details>

### Задача 36. Attention руками
Посчитайте выход attention ($d_k = 2$) для

```math
Q = \begin{pmatrix}1 & 0\end{pmatrix}, \quad K = \begin{pmatrix}1 & 0\\ 0 & 1\\ 1 & 1\end{pmatrix}, \quad V = \begin{pmatrix}1 & 2\\ 3 & 4\\ 5 & 6\end{pmatrix}
```

<details><summary>▶️ Решение</summary>

Скоры $QK^\top = (1, 0, 1)$, делим на $\sqrt{2}$: $(0.707, 0, 0.707)$. Softmax: $e^{0.707} = 2.03$, сумма $2.03 + 1 + 2.03 = 5.06$ → веса $(0.401, 0.198, 0.401)$. Выход: $0.401(1,2) + 0.198(3,4) + 0.401(5,6) = (3.0, 4.0)$.
</details>

### Задача 37. Число параметров и память LLM
Модель 8B параметров. Сколько памяти нужно для: (а) инференса в BF16, (б) INT4, (в) полного обучения с AdamW в смешанной точности?

<details><summary>▶️ Решение</summary>

(а) 8B × 2 байта = 16 ГБ + KV-кэш. (б) 8B × 0.5 = 4 ГБ (+ масштабы групп, ~4.5 ГБ). (в) BF16-веса 2 + BF16-градиенты 2 + FP32 мастер-веса 4 + Adam m и v по 4 = **16 байт на параметр** → 128 ГБ без активаций — не помещается в одну 80-ГБ карту, нужен ZeRO/FSDP. LoRA: базовые веса замороженные (16 ГБ) + крошечное состояние адаптеров.
</details>

### Задача 38. Дисперсия при dropout
Как dropout с $p = 0.5$ (inverted) меняет матожидание и дисперсию активации $h$?

<details><summary>▶️ Решение</summary>

$\tilde{h} = h\cdot m/(1-p)$, $m \sim \text{Bernoulli}(1-p)$. $\mathbb{E}[\tilde h] = h$. $\mathbb{E}[\tilde h^2] = h^2/(1-p)$ ⇒ $\mathrm{Var}[\tilde h] = h^2\frac{p}{1-p} = h^2$ при $p = 0.5$. Дисперсия на обучении выше, чем на инференсе, — одна из причин, почему dropout плохо сочетается с BatchNorm (статистики BN смещаются).
</details>

---

<a id="p-gen"></a>

## ✨ LLM и генеративные

### Задача 39. Speculative decoding
Черновая модель предлагает токен с вероятностью $q(x)$, целевая даёт $p(x)$. Принимаем с вероятностью $\min(1, p/q)$, иначе сэмплируем из $\mathrm{norm}(\max(0, p - q))$. Проверьте, что итоговое распределение — ровно $p$.

<details><summary>▶️ Решение</summary>

```python
rng = np.random.default_rng(0)
p = np.array([0.5, 0.3, 0.2]); q = np.array([0.2, 0.2, 0.6])
resid = np.maximum(p - q, 0); resid /= resid.sum()
out = []
for _ in range(200_000):
    x = rng.choice(3, p=q)
    out.append(x if rng.random() < min(1, p[x] / q[x]) else rng.choice(3, p=resid))
print(np.bincount(out) / len(out))   # ≈ [0.5, 0.3, 0.2]
```
Аналитически: $P(x) = q(x)\min(1, p/q) + P(\text{reject})\cdot\text{resid}(x) = \min(p, q) + (p - \min(p, q)) = p(x)$, поскольку $P(\text{reject}) = \sum_x \max(0, p - q)$.
</details>

### Задача 40. Шаг диффузии
Реализуйте функцию зашумления $q(\mathbf{x}_t \mid \mathbf{x}_0)$ и лосс DDPM для одного батча (сеть — заглушка).

<details><summary>▶️ Решение</summary>

```python
import torch
T = 1000
betas = torch.linspace(1e-4, 0.02, T); abar = torch.cumprod(1 - betas, 0)

def ddpm_loss(model, x0):
    t = torch.randint(0, T, (x0.shape[0],))
    eps = torch.randn_like(x0)
    a = abar[t].view(-1, *[1] * (x0.dim() - 1))
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return torch.nn.functional.mse_loss(model(xt, t), eps)

dummy = lambda x, t: torch.zeros_like(x)
print(ddpm_loss(dummy, torch.randn(8, 3, 32, 32)))   # ≈ 1.0: предсказание «нулевого шума» даёт дисперсию ε
```
</details>

---

<a id="p-easy"></a>

## 🟢 Простые задачи для разминки (41–50)

> Для тех, кто только начинает: хватит модулей 00½, 01 и начала 02–06. Решаются на бумаге за 5–10 минут.

### Задача 41. Порядки величин
Модель на 13B параметров обучили на 2T токенов. Сколько FLOP заняло обучение? Сколько дней на кластере с суммарной производительностью $10^{17}$ FLOP/с при утилизации 40%?

<details><summary>▶️ Решение</summary>

$6ND = 6 \cdot 13 \cdot 10^9 \cdot 2 \cdot 10^{12} \approx 1.56 \cdot 10^{23}$ FLOP. Эффективная скорость $4 \cdot 10^{16}$ FLOP/с → $3.9 \cdot 10^6$ с ≈ **45 дней**.
</details>

### Задача 42. Скалярное произведение и угол
$\mathbf{a} = (1, 2, 2)$, $\mathbf{b} = (2, 0, 1)$. Найдите $\mathbf{a}\cdot\mathbf{b}$, длины, косинус угла.

<details><summary>▶️ Решение</summary>

$\mathbf{a}\cdot\mathbf{b} = 2 + 0 + 2 = 4$; $\lVert\mathbf{a}\rVert = 3$, $\lVert\mathbf{b}\rVert = \sqrt5 \approx 2.236$; $\cos\theta = 4/(3\cdot2.236) \approx 0.596$, угол ≈ 53°.
</details>

### Задача 43. Умножение матриц руками

```math
\begin{pmatrix}1 & 2\\ 0 & 1\end{pmatrix}\begin{pmatrix}3 & 0\\ 1 & 2\end{pmatrix} = ?
```

<details><summary>▶️ Решение</summary>

Строка × столбец:

```math
\begin{pmatrix}1\cdot3 + 2\cdot1 & 1\cdot0 + 2\cdot2\\ 0\cdot3 + 1\cdot1 & 0\cdot0 + 1\cdot2\end{pmatrix} = \begin{pmatrix}5 & 4\\ 1 & 2\end{pmatrix}
```

Проверка: `np.array([[1,2],[0,1]]) @ np.array([[3,0],[1,2]])`.
</details>

### Задача 44. Производные
Найдите производные: $f(x) = 3x^2 - 4x + 1$, $g(x) = e^{2x}$, $h(x) = \ln(1 + x^2)$, $s(x) = \sigma(3x)$.

<details><summary>▶️ Решение</summary>

$f' = 6x - 4$; $g' = 2e^{2x}$; $h' = \frac{2x}{1 + x^2}$; $s' = 3\,\sigma(3x)(1 - \sigma(3x))$ — во всех трёх последних работает цепное правило.
</details>

### Задача 45. Один шаг градиентного спуска
$L(w) = (2w - 4)^2$, $w_0 = 0$, $\eta = 0.1$. Сделайте два шага.

<details><summary>▶️ Решение</summary>

$L'(w) = 4(2w - 4)$. $L'(0) = -16 \Rightarrow w_1 = 1.6$. $L'(1.6) = -3.2 \Rightarrow w_2 = 1.92$. Минимум — $w = 2$; расстояние до него уменьшается в 5 раз за шаг ($\lvert 1 - 0.1\cdot8\rvert = 0.2$).
</details>

### Задача 46. Среднее, медиана, дисперсия
Задержки ответа API (мс): 20, 22, 25, 21, 400. Посчитайте среднее, медиану, стандартное отклонение. Какую метрику показывать на дашборде?

<details><summary>▶️ Решение</summary>

Среднее = 97.6; медиана = 22; σ ≈ 151 (по n). Один выброс «утянул» среднее в 4 раза. Для задержек показывают **перцентили**: p50 (медиана), p95, p99 — они описывают и типичный случай, и хвост.
</details>

### Задача 47. Вероятность «хотя бы один»
Модель ошибается на 2% запросов независимо. Какова вероятность хотя бы одной ошибки на 50 запросах? А на 500?

<details><summary>▶️ Решение</summary>

$1 - 0.98^{50} \approx 0.636$; $1 - 0.98^{500} \approx 0.99996$. Вывод для агентов на LLM: цепочка из многих шагов ломается почти наверняка, даже если каждый шаг надёжен на 98%.
</details>

### Задача 48. Softmax руками
Логиты $(2, 1, 0)$. Найдите вероятности и CE-лосс, если правильный класс — второй.

<details><summary>▶️ Решение</summary>

$e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$, сумма 11.107 → $(0.665, 0.245, 0.090)$. Лосс $-\ln 0.245 \approx 1.41$.
</details>

### Задача 49. Матрица ошибок
Тест на 200 пациентах: 30 больных, тест нашёл 24 из них и ещё дал 16 ложных тревог. Precision, recall, accuracy?

<details><summary>▶️ Решение</summary>

TP = 24, FN = 6, FP = 16, TN = 154. Precision = 24/40 = 0.6, recall = 24/30 = 0.8, accuracy = 178/200 = 0.89.
</details>

### Задача 50. Нормализация признаков
Признаки «возраст» (20–70) и «доход» (20 000–300 000). Зачем и как их привести к одному масштабу?

<details><summary>▶️ Решение</summary>

Без нормализации доход доминирует в расстояниях (kNN, k-means), регуляризация штрафует признаки неравномерно, а у градиентного спуска вытянутые линии уровня (модуль 05). Стандартизация $z = (x - \mu)/\sigma$ — **с μ и σ, посчитанными только на train** (иначе утечка), затем те же значения применяются к val/test.
</details>

---

<a id="p-adv"></a>

## 🔴 Продвинутые задачи (51–60)

> Уровень Senior / Research: на стыке модулей 11–17.

### Задача 51. Градиент через attention
Для одного запроса $\mathbf{a} = \mathrm{softmax}(K\mathbf{q}/\sqrt{d})$, $\mathbf{o} = V^\top\mathbf{a}$. Найдите $\partial\mathbf{o}/\partial\mathbf{q}$ и объясните, когда градиент по запросу исчезает.

<details><summary>▶️ Решение</summary>

```math
\frac{\partial \mathbf{o}}{\partial \mathbf{q}} = \frac{1}{\sqrt d}\, V^\top\bigl(\mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top\bigr) K
```
Якобиан softmax $\mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top$ → 0, когда $\mathbf{a}$ почти one-hot (насыщение). Именно поэтому делят на $\sqrt d$; в больших моделях для стабильности дополнительно нормируют q и k (QK-norm) или ограничивают логиты (soft-capping).
</details>

### Задача 52. Сложность и память FlashAttention
Почему FlashAttention не уменьшает число FLOP, но ускоряет attention в 2–4 раза?

<details><summary>▶️ Решение</summary>

Узкое место стандартной реализации — не арифметика, а **чтение/запись матрицы $T\times T$ в HBM** (видеопамять). FlashAttention считает по блокам в быстрой SRAM, поддерживая онлайн-softmax (текущий максимум и сумму, модуль 13), и никогда не материализует $T\times T$: память $O(T)$ вместо $O(T^2)$, обращений к HBM в разы меньше. FLOP даже немного больше (пересчёт на backward), но ядро ограничено пропускной способностью памяти, а не вычислениями.
</details>

### Задача 53. Квантизация INT8
Реализуйте симметричную поканальную квантизацию весов в INT8 и оцените ошибку. Что будет, если в канале есть выброс?

<details><summary>▶️ Решение</summary>

```python
W = np.random.randn(256, 512).astype(np.float32)
W[3, 7] = 40.0                                        # выброс в канале 3
def quant(W, per_channel=True):
    m = np.abs(W).max(axis=1, keepdims=True) if per_channel else np.abs(W).max()
    s = m / 127
    return np.clip(np.round(W / s), -127, 127) * s
for pc in [False, True]:
    E = quant(W, pc) - W
    print("поканально" if pc else "на тензор ", np.abs(E).mean().round(4), np.abs(E[3]).mean().round(4))
```
Один выброс растягивает масштаб: при квантизации на тензор шаг сетки огромный для **всех** весов; поканально страдает только канал 3. Отсюда групповая квантизация (группы по 64–128) и методы борьбы с выбросами в LLM (AWQ, SmoothQuant, вращения).
</details>

### Задача 54. KV-кэш и GQA
LLaMA-подобная модель: 80 слоёв, 64 головы запросов по 128, контекст 128k, BF16. Сколько памяти занимает KV-кэш на одну последовательность при MHA и при GQA с 8 KV-головами?

<details><summary>▶️ Решение</summary>

MHA: $2 \cdot 80 \cdot 131072 \cdot 64 \cdot 128 \cdot 2$ байт ≈ **344 ГБ**. GQA (8 KV-голов): в 8 раз меньше ≈ **43 ГБ**. Поэтому длинный контекст без GQA/MQA, квантизации кэша (FP8/INT4) или MLA (сжатие KV в латент, DeepSeek-V2/V3) практически невозможен.
</details>

### Задача 55. Policy gradient на бандите
Обучите softmax-политику на 3-руком бандите методом REINFORCE с baseline и без. Сравните дисперсию оценок градиента.

<details><summary>▶️ Решение</summary>

```python
rng = np.random.default_rng(0)
mu = np.array([1.0, 1.5, 2.0]) + 10                   # большие награды → большая дисперсия без baseline
def grad_samples(theta, baseline, n=2000):
    p = np.exp(theta - theta.max()); p /= p.sum()
    a = rng.choice(3, n, p=p); r = mu[a] + rng.normal(size=n)
    glogp = np.eye(3)[a] - p                          # ∇θ log softmax
    return glogp * (r - baseline)[:, None]
theta = np.zeros(3)
for b in [0.0, 11.5]:
    g = grad_samples(theta, b)
    print(f"baseline={b:5}: среднее {g.mean(0).round(3)}, дисперсия {g.var(0).sum():.2f}")
```
Средние градиенты совпадают с точностью до шума (baseline не смещает оценку), а дисперсия с baseline ≈ средней наградой падает на два порядка.
</details>

### Задача 56. Спектральный радиус и RNN
Линейная RNN $\mathbf{h}_t = W\mathbf{h}_{t-1}$. При каких собственных числах $W$ градиент через 100 шагов не затухает и не взрывается? Как это используют SSM (Mamba, S4)?

<details><summary>▶️ Решение</summary>

$\mathbf{h}_{100} = W^{100}\mathbf{h}_0$, по собственным направлениям множители $\lambda_i^{100}$: нужно $\lvert\lambda_i\rvert \approx 1$. Обычная RNN этого не гарантирует. SSM параметризуют переходную матрицу диагональной с $\lambda = e^{-\Delta\cdot a}$, $a > 0$: всегда $\lvert\lambda\rvert < 1$ (устойчиво), а обучаемый шаг $\Delta$ (в Mamba — зависящий от входа) управляет длиной памяти от нескольких до тысяч шагов. Линейность позволяет считать рекуррентность параллельным сканированием на обучении.
</details>

### Задача 57. Ранг обновлений при дообучении
Проверьте гипотезу LoRA: возьмите разность двух матриц «до/после» с низкоранговым сигналом и шумом и найдите, сколько сингулярных чисел несут 90% энергии.

<details><summary>▶️ Решение</summary>

```python
d, r_true = 512, 8
dW = np.random.randn(d, r_true) @ np.random.randn(r_true, d) + 0.3 * np.random.randn(d, d)
S = np.linalg.svd(dW, compute_uv=False)
energy = np.cumsum(S**2) / np.sum(S**2)
print(np.searchsorted(energy, 0.9) + 1, "из", d)
```
В реальных дообучениях LLM спектр $\Delta W$ быстро убывает — поэтому LoRA с r = 8–64 достигает качества полного дообучения на многих задачах. Для задач, сильно меняющих знания модели (новый язык, домен), низкого ранга бывает недостаточно.
</details>

### Задача 58. ELBO = log p − KL
Докажите тождество $\log p(\mathbf{x}) = \text{ELBO} + D_{\mathrm{KL}}\bigl(q(\mathbf{z}\mid\mathbf{x}) \parallel p(\mathbf{z}\mid\mathbf{x})\bigr)$.

<details><summary>▶️ Решение</summary>

```math
\begin{aligned}
\log p(\mathbf{x}) &= \mathbb{E}_q\bigl[\log p(\mathbf{x})\bigr] = \mathbb{E}_q\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z}\mid\mathbf{x})}\right] = \mathbb{E}_q\left[\log\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}\mid\mathbf{x})}\right] + \mathbb{E}_q\left[\log\frac{q(\mathbf{z}\mid\mathbf{x})}{p(\mathbf{z}\mid\mathbf{x})}\right]\\
&= \underbrace{\mathbb{E}_q[\log p(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q \parallel p(\mathbf{z}))}_{\text{ELBO}} + D_{\mathrm{KL}}\bigl(q \parallel p(\mathbf{z}\mid\mathbf{x})\bigr)
\end{aligned}
```
KL ≥ 0 ⇒ ELBO — нижняя оценка; максимизация ELBO одновременно учит декодер и приближает энкодер к истинному апостериорному.
</details>

### Задача 59. Calibration через temperature scaling
Модель переуверена. Подберите температуру $T$ на валидации, минимизируя NLL, и покажите, что accuracy не меняется.

<details><summary>▶️ Решение</summary>

```python
from scipy.optimize import minimize_scalar
rng = np.random.default_rng(0)
N, K = 5000, 10
y = rng.integers(0, K, N)
logits = rng.normal(size=(N, K)); logits[np.arange(N), y] += 1.5
logits *= 3                                            # искусственная переуверенность
def nll(T):
    z = logits / T; z = z - z.max(1, keepdims=True)
    return -(z[np.arange(N), y] - np.log(np.exp(z).sum(1))).mean()
T = minimize_scalar(nll, bounds=(0.1, 10), method="bounded").x
print(round(T, 2), nll(1.0).round(3), nll(T).round(3))   # T ≈ 2: NLL падает с 1.72 до 1.42
```
Деление всех логитов на одно $T > 0$ не меняет argmax → accuracy та же, меняется только уверенность.
</details>

### Задача 60. Оценка спектральной нормы для Липшица
Сеть $f = W_3\,\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1\mathbf{x}))$. Дайте верхнюю оценку константы Липшица и объясните связь со spectral normalization и робастностью.

<details><summary>▶️ Решение</summary>

ReLU 1-липшицева, линейный слой — с константой $\lVert W\rVert_2 = \sigma_{\max}(W)$. Композиция: $\mathrm{Lip}(f) \le \lVert W_3\rVert_2\lVert W_2\rVert_2\lVert W_1\rVert_2$. Spectral normalization делит каждый $W$ на $\sigma_{\max}$ (степенной метод, задача 3) → $\mathrm{Lip} \le 1$: так ограничивают критик WGAN (модуль 17) и получают гарантии устойчивости к $L_2$-возмущениям входа. Оценка часто очень грубая — реальная константа может быть на порядки меньше произведения норм.
</details>

---

<a id="p-bugs"></a>

## 🐞 Найди баг (61–70)

> Код запускается без ошибок, но считает **неправильно** — самые опасные баги в ML. Найдите ошибку, прежде чем открывать решение. Каждое решение заканчивается проверкой, которая ловит баг.

### Задача 61. Softmax, который иногда возвращает NaN
```py
def softmax(z):
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

<details><summary>▶️ Решение</summary>

Переполнение `exp` при больших логитах (модуль 13). Вычитаем максимум по той же оси:

```python
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
p = softmax(np.array([[1000., 1001.], [1., 2.]]))
assert np.all(np.isfinite(p)) and np.allclose(p.sum(1), 1)
```
</details>

### Задача 62. Стандартизация по батчу
```py
X = np.random.randn(256, 10) * 5 + 3          # 256 объектов, 10 признаков
X_norm = (X - X.mean()) / X.std()
```

<details><summary>▶️ Решение</summary>

Среднее и std посчитаны по **всей матрице**, а нужно по каждому признаку (ось объектов `axis=0`). Если признаки в разных масштабах, нормировка не работает.

```python
X = np.random.randn(256, 10) * np.arange(1, 11) + 3
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
assert np.allclose(X_norm.mean(0), 0, atol=1e-10) and np.allclose(X_norm.std(0), 1)
```
</details>

### Задача 63. Кросс-энтропия по неправильной оси
```py
logits = np.random.randn(32, 5); y = np.random.randint(0, 5, 32)
p = softmax(logits)
loss = -np.mean(np.log(p[y, np.arange(32)]))
```

<details><summary>▶️ Решение</summary>

Индексы перепутаны местами: `p[y, arange]` берёт строку по метке и столбец по номеру объекта (а при 32 > 5 ещё и падает с IndexError — повезло, что падает). Правильно — `p[объект, метка]`. Ещё лучше — считать через log-softmax, без `log(softmax)`:

```python
def cross_entropy(logits, y):
    z = logits - logits.max(1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -log_p[np.arange(len(y)), y].mean()
logits = np.random.randn(32, 5); y = np.random.randint(0, 5, 32)
assert np.isclose(cross_entropy(logits, y), -np.mean(np.log(softmax(logits)[np.arange(32), y])))
```
</details>

### Задача 64. Утечка при нормировке
```py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_scaled = StandardScaler().fit_transform(X)
X_tr, X_te = train_test_split(X_scaled, test_size=0.2)
```

<details><summary>▶️ Решение</summary>

Среднее и дисперсия посчитаны и по тесту — информация из теста «утекла» в обучение. Для нормировки эффект небольшой, но тот же паттерн с отбором признаков, таргет-энкодингом или PCA даёт сильно завышенные метрики. `fit` — только на train:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = np.random.randn(500, 4)
X_tr, X_te = train_test_split(X, test_size=0.2, random_state=0)
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
assert np.allclose(X_tr.mean(0), 0, atol=1e-10)            # среднее теста не обязано быть 0
```
Надёжный способ — `sklearn.pipeline.Pipeline`: внутри кросс-валидации он сам делает `fit` только на обучающих фолдах.
</details>

### Задача 65. Градиентный спуск, который уходит вверх
```py
w = np.zeros(3); X = np.random.randn(100, 3); y = X @ np.array([1., -2., .5])
for _ in range(500):
    grad = X.T @ (y - X @ w) / len(y)
    w -= 0.1 * grad
```

<details><summary>▶️ Решение</summary>

Знак градиента: для $L = \frac{1}{2N}\lVert X\mathbf{w} - \mathbf{y}\rVert^2$ градиент равен $X^\top(X\mathbf{w} - \mathbf{y})/N$, а в коде $X^\top(\mathbf{y} - X\mathbf{w})$ — шаг идёт в сторону роста лосса.

```python
w = np.zeros(3); X = np.random.randn(100, 3); y = X @ np.array([1., -2., .5])
for _ in range(500):
    w -= 0.1 * X.T @ (X @ w - y) / len(y)
assert np.allclose(w, [1, -2, .5], atol=1e-3)
```
Привычка: всегда проверять, что лосс **уменьшается** на первых шагах, и сверять градиент численно (модуль 03).
</details>

### Задача 66. Adam без коррекции смещения
```py
m = v = 0
for t in range(1, 6):
    g = 1.0                                   # постоянный градиент
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g**2
    step = m / (np.sqrt(v) + 1e-8)
    print(t, round(step, 3))
```

<details><summary>▶️ Решение</summary>

Без деления на $1 - \beta^t$ первые шаги искажены: $m_1 = 0.1$, $v_1 = 0.001$, шаг $\approx 0.1/0.0316 \approx 3.16$ — в 3 раза больше «нормального» шага 1. С коррекцией шаг сразу правильный:

```python
m = v = 0
for t in range(1, 6):
    g = 1.0
    m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g**2
    step = (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    assert np.isclose(step, 1.0)
```
</details>

### Задача 67. Dropout на инференсе
```py
def dropout(h, p=0.5, training=True):
    mask = np.random.rand(*h.shape) > p
    return h * mask                           # и на обучении, и на инференсе
```

<details><summary>▶️ Решение</summary>

Две ошибки: нет масштабирования на $1/(1-p)$ — матожидание активаций на обучении вдвое меньше, чем на инференсе; и маска применяется всегда. Inverted dropout:

```python
def dropout(h, p=0.5, training=True):
    if not training or p == 0:
        return h
    mask = np.random.rand(*h.shape) > p
    return h * mask / (1 - p)
h = np.ones((1000, 1000))
assert abs(dropout(h).mean() - 1) < 0.01 and np.array_equal(dropout(h, training=False), h)
```
</details>

### Задача 68. Несмещённая дисперсия, которая смещена
```py
def sample_std(x):
    return np.std(x)                          # для доверительного интервала по маленькой выборке
```

<details><summary>▶️ Решение</summary>

`np.std` по умолчанию делит на $n$ (`ddof=0`) — оценка дисперсии смещена вниз в $\frac{n-1}{n}$ раз; при $n = 5$ это −20%. Для оценки по выборке — `ddof=1` (так считает `pandas.Series.std`, а `numpy` — нет).

```python
rng = np.random.default_rng(0)
samples = rng.normal(0, 1, (200_000, 5))
print(samples.var(axis=1, ddof=0).mean().round(3), samples.var(axis=1, ddof=1).mean().round(3))   # 0.8 и 1.0
```
</details>

### Задача 69. Косинусное сходство без нормировки
```py
def top_k(query, docs, k=3):
    return np.argsort(-(docs @ query))[:k]    # «ищем самые похожие по косинусу»
```

<details><summary>▶️ Решение</summary>

Это скалярное произведение, а не косинус: длинные векторы (часто — длинные или «шумные» документы) выигрывают только из-за нормы.

```python
def top_k(query, docs, k=3):
    d = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    q = query / np.linalg.norm(query)
    return np.argsort(-(d @ q))[:k]
docs = np.array([[1., 0.], [10., 9.], [0.7, 0.1]])
print(np.argsort(-(docs @ np.array([1., 0.])))[:1], top_k(np.array([1., 0.]), docs, 1))   # [1] против [0]
```
</details>

### Задача 70. Accuracy 99% на несбалансированных данных
```py
y_true = np.r_[np.zeros(990), np.ones(10)]
y_pred = model.predict(X)                     # модель всегда предсказывает 0
print((y_pred == y_true).mean())              # 0.99 — «отличная модель»
```

<details><summary>▶️ Решение</summary>

Баг в выборе метрики, а не в коде: при 1% позитивов константа даёт 99% accuracy и нулевую пользу (модуль 09). Смотрите recall, precision, PR-AUC и сравнивайте с бейзлайном-константой.

```python
y_true = np.r_[np.zeros(990), np.ones(10)]; y_pred = np.zeros(1000)
tp = np.sum((y_pred == 1) & (y_true == 1))
recall = tp / y_true.sum()
print((y_pred == y_true).mean(), recall)      # 0.99 и 0.0
```
</details>

---

<a id="p-fermi"></a>

## 🧮 Оценка на салфетке (71–80)

> На собеседованиях Senior и в реальной работе постоянно нужно за минуту прикинуть порядок величины. Точный ответ не важен — важен ход рассуждения и допущения. Решайте без калькулятора.

### Задача 71. Сколько стоит обучение
Модель 8B параметров, 15T токенов. Видеокарта даёт ~400 TFLOP/s в BF16 при утилизации 40%. Сколько GPU-часов? Сколько стоит при цене 2 USD за GPU-час?

<details><summary>▶️ Решение</summary>

$6ND = 6 \cdot 8 \cdot 10^9 \cdot 15 \cdot 10^{12} = 7.2 \cdot 10^{23}$ FLOP. Эффективно $1.6 \cdot 10^{14}$ FLOP/с на карту → $4.5 \cdot 10^9$ с ≈ **1.25 млн GPU-часов** ≈ **2.5 млн USD**. На 1000 картах — ~52 дня.
</details>

### Задача 72. Стоимость инференса
Модель 70B, генерация 1 млн токенов. Сколько FLOP? Если карта выдаёт эффективно $10^{14}$ FLOP/с, сколько GPU-секунд?

<details><summary>▶️ Решение</summary>

На токен ≈ $2N = 1.4 \cdot 10^{11}$ FLOP; на миллион — $1.4 \cdot 10^{17}$; при $10^{14}$ FLOP/с — 1400 GPU-секунд ≈ **0.4 GPU-часа**. На практике генерация упирается не в FLOP, а в чтение весов из памяти: 140 ГБ весов на каждый шаг декодирования — поэтому батчинг запросов кратно удешевляет инференс.
</details>

### Задача 73. Память векторного индекса
100 млн чанков, эмбеддинги 1024 float32. Сколько памяти? А с PQ по 64 байта на вектор?

<details><summary>▶️ Решение</summary>

$10^8 \cdot 1024 \cdot 4 = 4.1 \cdot 10^{11}$ байт ≈ **410 ГБ**. PQ: $10^8 \cdot 64 = 6.4$ ГБ — влезает в память одного сервера. Плюс граф HNSW: ~32 соседа × 4 байта × $10^8$ ≈ 13 ГБ.
</details>

### Задача 74. Сколько текста в датасете
Сколько токенов в 10 млн веб-страниц средней длиной 5 КБ текста?

<details><summary>▶️ Решение</summary>

Английский текст: ~4 символа (байта) на токен → 5 КБ ≈ 1250 токенов. Итого ≈ $1.25 \cdot 10^{10}$ = **12.5B токенов**. Для русского в UTF-8 (2 байта на букву) и токенизаторе, обученном в основном на английском, токенов на тот же объём текста заметно больше.
</details>

### Задача 75. Хватит ли трафика на A/B-тест
Конверсия 3%, хотим заметить относительный рост на 5%. Сайт — 20 000 посетителей в день. Сколько дней тест?

<details><summary>▶️ Решение</summary>

$\delta = 0.0015$; $n \approx 16 \cdot 0.03 \cdot 0.97 / 0.0015^2 \approx 207\,000$ на группу, 414 000 всего → **~21 день**. Маленькие эффекты на маленьких конверсиях требуют огромных выборок; варианты — метрика ближе к действию, CUPED, больше трафика.
</details>

### Задача 76. Время одной эпохи
ImageNet 1.28 млн картинок, ResNet-50 ≈ 4 GFLOP на прямой проход. Обучение ≈ 3× прямого. Карта эффективно $10^{14}$ FLOP/с. Сколько длится эпоха?

<details><summary>▶️ Решение</summary>

$1.28 \cdot 10^6 \cdot 4 \cdot 10^9 \cdot 3 \approx 1.5 \cdot 10^{16}$ FLOP → 150 с на одной карте в идеале. На практике упирается в загрузку и аугментацию данных — реальные минуты на эпоху; узкое место часто CPU, а не GPU.
</details>

### Задача 77. KV-кэш и число одновременных пользователей
Карта 80 ГБ, модель 8B в BF16 (16 ГБ). KV-кэш — 128 КБ на токен. Сколько пользователей с контекстом 8K поместится?

<details><summary>▶️ Решение</summary>

Свободно ~60 ГБ (оставляем запас на активации). На пользователя $8192 \cdot 128$ КБ = 1 ГБ → **~60 одновременных сессий**. Поэтому важны GQA, квантизация KV-кэша и paged attention (vLLM), чтобы не резервировать память под максимальную длину.
</details>

### Задача 78. Сколько градиентных шагов
Предобучение 2T токенов, батч 4M токенов. Сколько шагов оптимизатора? Сколько на warmup 1%?

<details><summary>▶️ Решение</summary>

$2 \cdot 10^{12} / 4 \cdot 10^6 = 500\,000$ шагов; warmup — 5000 шагов.
</details>

### Задача 79. Стоимость разметки
Нужно 50 000 размеченных пар «вопрос — релевантный документ». Асессор делает 60 пар в час, 15% меток проверяет второй асессор, ставка 10 USD/ч. Бюджет?

<details><summary>▶️ Решение</summary>

$50\,000 / 60 \approx 833$ часа + 15% проверки ≈ 958 часов → **≈ 9600 USD**. Альтернатива — сгенерировать вопросы LLM по документам и разметить вручную только выборку для контроля качества.
</details>

### Задача 80. Точность оценки модели
На тестовом наборе из 2000 примеров accuracy 85%. Насколько можно доверять разнице с конкурентом, у которого 86%?

<details><summary>▶️ Решение</summary>

SE $= \sqrt{0.85 \cdot 0.15 / 2000} \approx 0.008$, 95%-интервал ±1.6 п.п. — разница в 1 п.п. в пределах шума. Если модели проверялись на **одном** наборе, корректнее парный тест (Макнемар, задача 26): он учитывает, на каких примерах ошибаются обе модели, и обычно чувствительнее.
</details>

---

<a id="p-projects"></a>

## 🛠 Мини-проекты с нуля (81–90)

> Реализуйте алгоритм на чистом NumPy за 20–40 строк и сверьте с эталоном из scikit-learn. Это лучший способ убедиться, что вы понимаете математику, а не только API.

### Задача 81. Дерево решений (критерий Джини)
Реализуйте классификационное дерево глубины ≤ 4 и сравните точность с `DecisionTreeClassifier` на датасете `breast_cancer`.

<details><summary>▶️ Решение</summary>

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def gini(y):
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - np.sum(p**2)

def best_split(X, y):
    best = (None, None, gini(y))                       # (признак, порог, взвешенный gini)
    for j in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, 19))):
            left = X[:, j] <= t
            if left.all() or not left.any():
                continue
            g = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if g < best[2]:
                best = (j, t, g)
    return best

def build(X, y, depth=0, max_depth=4):
    j, t, _ = best_split(X, y)
    if depth == max_depth or j is None or len(y) < 10:
        return np.bincount(y, minlength=2).argmax()     # лист: мажоритарный класс
    left = X[:, j] <= t
    return (j, t, build(X[left], y[left], depth + 1, max_depth), build(X[~left], y[~left], depth + 1, max_depth))

def predict_one(node, x):
    while isinstance(node, tuple):
        j, t, l, r = node
        node = l if x[j] <= t else r
    return node

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
tree = build(Xtr, ytr)
mine = np.mean([predict_one(tree, x) == t for x, t in zip(Xte, yte)])
ref = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr).score(Xte, yte)
print(round(mine, 3), round(ref, 3))                     # ≈ 0.92 и ≈ 0.95 — почти как у sklearn
```
Отличие от sklearn: мы перебираем 19 квантилей вместо всех порогов — быстрее и почти без потери качества (так же устроены гистограммные бустинги LightGBM и HistGradientBoosting).
</details>

### Задача 82. Градиентный бустинг на «пеньках»
Реализуйте бустинг для регрессии: каждый шаг — дерево глубины 1, подогнанное к остаткам (антиградиенту MSE).

<details><summary>▶️ Решение</summary>

```python
def fit_stump(x, r):
    best = (np.inf, None, None, None)
    for t in np.quantile(x, np.linspace(0.02, 0.98, 49)):
        l = x <= t
        if l.all() or not l.any():
            continue
        pl, pr = r[l].mean(), r[~l].mean()
        err = np.sum((r[l] - pl) ** 2) + np.sum((r[~l] - pr) ** 2)
        if err < best[0]:
            best = (err, t, pl, pr)
    return best[1:]

rng = np.random.default_rng(0)
x = rng.uniform(0, 6, 400); y = np.sin(x) + 0.2 * rng.normal(size=400)
F = np.full_like(y, y.mean()); lr = 0.1; stumps = []
for m in range(200):
    t, pl, pr = fit_stump(x, y - F)                     # остатки = −∇ MSE
    F += lr * np.where(x <= t, pl, pr); stumps.append((t, pl, pr))
print(round(np.mean((y - F) ** 2), 4))                  # ≈ шум 0.04 — модель выучила sin
```
Каждый «пенёк» — шаг градиентного спуска в пространстве функций; `lr` — learning rate (модуль 10).
</details>

### Задача 83. k-means++
Реализуйте инициализацию k-means++ и покажите, что она даёт меньшую инерцию, чем случайная, в среднем по запускам.

<details><summary>▶️ Решение</summary>

```python
from sklearn.datasets import make_blobs
X, _ = make_blobs(1500, centers=8, cluster_std=0.8, random_state=1)

def kmeans(X, C, iters=30):
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(len(C))])
    return ((X - C[lab]) ** 2).sum()

def init_pp(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None] - np.array(C)[None]) ** 2).sum(-1).min(1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])   # дальние точки — вероятнее
    return np.array(C)

rng = np.random.default_rng(0)
rand = [kmeans(X, X[rng.choice(len(X), 8, replace=False)]) for _ in range(20)]
pp = [kmeans(X, init_pp(X, 8, rng)) for _ in range(20)]
print(round(np.mean(rand)), round(np.mean(pp)))         # k-means++ в среднем лучше
```
</details>

### Задача 84. PCA на рукописных цифрах
Сожмите изображения 8×8 (64 признака) с помощью PCA: сколько компонент объясняют 90% дисперсии? Какова ошибка реконструкции?

<details><summary>▶️ Решение</summary>

```python
from sklearn.datasets import load_digits
X = load_digits().data
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ratio = np.cumsum(S**2) / np.sum(S**2)
k = np.searchsorted(ratio, 0.9) + 1
X_rec = Xc @ Vt[:k].T @ Vt[:k] + X.mean(0)
print(k, round(np.mean((X - X_rec) ** 2) / X.var(), 3))  # 21 компонента из 64, ошибка ≈ 5% дисперсии
```
</details>

### Задача 85. Наивный Байес для текстов
Реализуйте мультиномиальный наивный Байес со сглаживанием Лапласа и проверьте на игрушечном спам-датасете.

<details><summary>▶️ Решение</summary>

```python
from collections import Counter
train = [("выиграй приз бесплатно сейчас", 1), ("бесплатно деньги приз", 1), ("срочно выиграй деньги", 1),
         ("встреча завтра в офисе", 0), ("отчёт по проекту завтра", 0), ("обсудим проект на встрече", 0)]
vocab = sorted({w for t, _ in train for w in t.split()})
counts = {c: Counter(w for t, y in train if y == c for w in t.split()) for c in (0, 1)}
prior = {c: np.mean([y == c for _, y in train]) for c in (0, 1)}

def log_posterior(text, c, alpha=1.0):
    total = sum(counts[c].values())
    return np.log(prior[c]) + sum(np.log((counts[c][w] + alpha) / (total + alpha * len(vocab)))
                                  for w in text.split() if w in vocab)

for text in ["бесплатно приз", "встреча по проекту"]:
    print(text, "→ спам" if log_posterior(text, 1) > log_posterior(text, 0) else "→ не спам")
```
Сглаживание $\alpha$ — это MAP с априором Дирихле (модули 07 и 22): без него одно незнакомое классу слово обнуляет вероятность.
</details>

### Задача 86. Токенизатор BPE
Реализуйте обучение Byte-Pair Encoding: повторять «найти самую частую пару соседних символов → склеить».

<details><summary>▶️ Решение</summary>

```python
from collections import Counter
corpus = "низкий низкий ниже новый новейший новые низко".split()
words = Counter(tuple(w) + ("</w>",) for w in corpus)
merges = []
for _ in range(10):
    pairs = Counter()
    for w, f in words.items():
        for a, b in zip(w, w[1:]):
            pairs[(a, b)] += f
    if not pairs:
        break
    best = max(pairs, key=pairs.get); merges.append(best)
    new = Counter()
    for w, f in words.items():
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                out.append(w[i] + w[i + 1]); i += 2
            else:
                out.append(w[i]); i += 1
        new[tuple(out)] += f
    words = new
print(merges[:5])
print(list(words)[:3])     # частые куски («низк», «нов») стали отдельными токенами
```
Так устроены токенизаторы GPT и LLaMA (на байтах, с десятками тысяч слияний). Отсюда и «математика токенов» из задачи 74.
</details>

### Задача 87. Биграммная языковая модель и perplexity
Обучите модель $p(c_t \mid c_{t-1})$ по символам и сравните perplexity со случайной моделью.

<details><summary>▶️ Решение</summary>

```python
text = ("градиентный спуск обновляет веса модели шаг за шагом пока функция потерь не перестанет "
        "уменьшаться и модель не начнёт хорошо предсказывать новые данные ") * 20
chars = sorted(set(text)); idx = {c: i for i, c in enumerate(chars)}; V = len(chars)
split = int(len(text) * 0.9); tr, te = text[:split], text[split:]
N = np.ones((V, V))                                         # сглаживание Лапласа
for a, b in zip(tr, tr[1:]):
    N[idx[a], idx[b]] += 1
P = N / N.sum(1, keepdims=True)
nll = -np.mean([np.log(P[idx[a], idx[b]]) for a, b in zip(te, te[1:])])
print(V, round(np.exp(nll), 2))                             # perplexity ≈ 6 против 29 у случайной модели
```
Perplexity — «из скольких символов в среднем выбирает модель» (модуль 08). Нейросетевые LM — это та же идея с длинным контекстом вместо одного символа.
</details>

### Задача 88. MLP на цифрах
Обучите двухслойную сеть на `load_digits` на NumPy (ReLU, softmax + CE, мини-батчи) и добейтесь точности > 95% на тесте.

<details><summary>▶️ Решение</summary>

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 64), (64, 128)); b1 = np.zeros(128)
W2 = rng.normal(0, np.sqrt(1 / 128), (128, 10)); b2 = np.zeros(10)
for epoch in range(30):
    batches = rng.permutation(len(Xtr))[: len(Xtr) // 32 * 32].reshape(-1, 32)   # перемешанные батчи по 32
    for i in batches:
        xb, yb = Xtr[i], ytr[i]
        a1 = xb @ W1 + b1; h = np.maximum(0, a1); z = h @ W2 + b2
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        dz = p.copy(); dz[np.arange(32), yb] -= 1; dz /= 32
        dW2 = h.T @ dz; db2 = dz.sum(0); dh = dz @ W2.T * (a1 > 0); dW1 = xb.T @ dh; db1 = dh.sum(0)
        for P_, G in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:
            P_ -= 0.1 * G
pred = (np.maximum(0, Xte @ W1 + b1) @ W2 + b2).argmax(1)
print(round((pred == yte).mean(), 3))                       # ≈ 0.97
```
</details>

### Задача 89. Логистическая регрессия с L1 через проксимальный шаг
Реализуйте ISTA (градиентный шаг + soft thresholding) и покажите, что часть весов становится ровно нулём.

<details><summary>▶️ Решение</summary>

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 20)); w_true = np.zeros(20); w_true[:3] = [2, -3, 1.5]
y = (rng.random(500) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
w = np.zeros(20); lr, lam = 0.5, 0.05
for _ in range(500):
    g = X.T @ (1 / (1 + np.exp(-X @ w)) - y) / len(y)
    w = w - lr * g
    w = np.sign(w) * np.maximum(np.abs(w) - lr * lam, 0)   # проксимальный оператор L1
print(np.round(w, 2)); print("нулевых весов:", np.sum(w == 0))
```
Обычный SGD с L1 колеблется около нуля, а проксимальный шаг даёт **точные нули** — это и есть отбор признаков Lasso (модуль 05, задача 11).
</details>

### Задача 90. Сравнение оптимизаторов на логистической регрессии
Сравните SGD, Momentum и Adam по лоссу после 200 шагов на плохо обусловленных признаках.

<details><summary>▶️ Решение</summary>

```python
rng = np.random.default_rng(1)
X = rng.normal(size=(1000, 10)) * np.logspace(0, 2, 10)       # масштабы признаков от 1 до 100
y = (X[:, 0] + 0.01 * X[:, -1] > 0).astype(float)
loss = lambda w: np.mean(np.logaddexp(0, X @ w) - y * (X @ w))
grad = lambda w: X.T @ (1 / (1 + np.exp(-np.clip(X @ w, -30, 30))) - y) / len(y)
res = {}
for name in ["SGD", "Momentum", "Adam"]:
    w = np.zeros(10); m = np.zeros(10); v = np.zeros(10)
    for t in range(1, 201):
        g = grad(w)
        if name == "SGD":
            w -= 1e-4 * g
        elif name == "Momentum":
            m = 0.9 * m + g; w -= 1e-4 * m
        else:
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g**2
            w -= 0.05 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    res[name] = round(loss(w), 4)
print(res)        # Adam заметно ниже: он выравнивает масштабы признаков
```
SGD и Momentum вынуждены брать крошечный шаг из-за признака с масштабом 100 и почти не двигаются по остальным. Нормировка признаков (задача 50) решила бы это и для них.
</details>

---

<a id="p-interview"></a>

## 🎤 Как на собеседовании (91–100)

> Задачи «на доске»: интервьюер ждёт рассуждение вслух. Попробуйте решить за 5–10 минут без кода, затем проверьте моделированием.

### Задача 91. Математическое ожидание максимума
$X, Y \sim U(0, 1)$ независимы. Найдите $\mathbb{E}[\max(X, Y)]$.

<details><summary>▶️ Решение</summary>

$P(\max \le t) = t^2$ ⇒ плотность $2t$ ⇒ $\mathbb{E} = \int_0^1 t \cdot 2t\,dt = 2/3$. Обобщение: максимум $n$ равномерных — $\frac{n}{n+1}$.

```python
print(np.random.rand(1_000_000, 2).max(1).mean())   # ≈ 0.667
```
</details>

### Задача 92. Монти Холл
Три двери, за одной приз. Вы выбрали дверь, ведущий открыл пустую из оставшихся и предложил сменить выбор. Менять?

<details><summary>▶️ Решение</summary>

Менять: вероятность выигрыша 2/3 против 1/3. Изначальный выбор верен с вероятностью 1/3, и ведущий эту вероятность не меняет; оставшиеся 2/3 «переходят» на единственную закрытую дверь.

```python
rng = np.random.default_rng(0); n = 100_000
prize = rng.integers(0, 3, n); pick = rng.integers(0, 3, n)
print((prize == pick).mean(), (prize != pick).mean())   # остаться ≈ 0.33, сменить ≈ 0.67
```
</details>

### Задача 93. Коллекционер купонов
Сколько в среднем нужно случайных выборок с возвращением из $n$ классов, чтобы увидеть каждый хотя бы раз? Сколько для $n = 100$?

<details><summary>▶️ Решение</summary>

Когда уже собрано $k$ классов, новый выпадает с вероятностью $\frac{n-k}{n}$ — ждём в среднем $\frac{n}{n-k}$ шагов. Сумма: $n\sum_{k=1}^{n}\frac{1}{k} = nH_n \approx n\ln n$. Для 100 — ≈ 519. Применение: сколько примеров нужно, чтобы в выборке встретился каждый из 100 редких классов.
</details>

### Задача 94. Докажите, что логистический лосс выпуклый
Покажите, что $\ell(z) = \log(1 + e^{-z})$ выпукла и что поэтому логистическая регрессия имеет единственный минимум (при регуляризации).

<details><summary>▶️ Решение</summary>

$\ell'(z) = -\sigma(-z)$, $\ell''(z) = \sigma(z)\sigma(-z) > 0$ — выпукла. Композиция выпуклой функции с линейной $z = y\,\mathbf{w}^\top\mathbf{x}$ выпукла, сумма выпуклых выпукла; гессиан $X^\top D X$ с $D = \mathrm{diag}(\sigma(1-\sigma)) \succeq 0$. Добавка $\lambda\lVert\mathbf{w}\rVert^2$ делает функцию строго выпуклой ⇒ минимум единственный. Без регуляризации на линейно разделимых данных минимума нет: веса уходят в бесконечность.
</details>

### Задача 95. Градиент нормировки вектора
Найдите якобиан $f(\mathbf{x}) = \mathbf{x}/\lVert\mathbf{x}\rVert$ (встречается в косинусном лоссе и QK-norm).

<details><summary>▶️ Решение</summary>

```math
\frac{\partial f}{\partial \mathbf{x}} = \frac{1}{\lVert\mathbf{x}\rVert}\left(I - \frac{\mathbf{x}\mathbf{x}^\top}{\lVert\mathbf{x}\rVert^2}\right)
```
Это проекция на плоскость, ортогональную $\mathbf{x}$: изменение вдоль самого $\mathbf{x}$ не меняет направления. Численная проверка:

```python
x = np.random.randn(5); n = np.linalg.norm(x)
J = (np.eye(5) - np.outer(x, x) / n**2) / n
h = 1e-6; J_num = np.array([((x + h * e) / np.linalg.norm(x + h * e) - (x - h * e) / np.linalg.norm(x - h * e)) / (2 * h) for e in np.eye(5)]).T
print(np.allclose(J, J_num, atol=1e-6))
```
</details>

### Задача 96. Случайное блуждание
Частица на прямой делает 100 шагов ±1 с равной вероятностью. Каково ожидаемое расстояние от старта (квадратичное)? Вероятность вернуться в 0 ровно на 100-м шаге?

<details><summary>▶️ Решение</summary>

$\mathbb{E}[S^2] = n = 100$ ⇒ среднеквадратичное расстояние $\sqrt{n} = 10$ (дисперсии независимых шагов складываются). $P(S_{100} = 0) = \binom{100}{50}/2^{100} \approx 0.08 \approx \sqrt{2/(\pi n)}$. Связь с ML: шум SGD накапливается как $\sqrt{t}$, а не как $t$.
</details>

### Задача 97. Ridge через байес
Покажите, что решение Ridge совпадает со средним апостериорного распределения весов при гауссовском шуме и гауссовском априоре.

<details><summary>▶️ Решение</summary>

$p(\mathbf{w} \mid D) \propto \exp\bigl(-\frac{1}{2\sigma^2}\lVert X\mathbf{w} - \mathbf{y}\rVert^2 - \frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2\bigr)$ — экспонента от квадратичной формы ⇒ гауссиана; её мода = среднее = минимум показателя: $(X^\top X + \lambda I)\mathbf{w} = X^\top\mathbf{y}$, $\lambda = \sigma^2/\tau^2$. Байесовский подход дополнительно даёт ковариацию $\sigma^2(X^\top X + \lambda I)^{-1}$ — неопределённость весов (модуль 22).
</details>

### Задача 98. Сколько раз подбросить монету
Как проверить, честная ли монета, с точностью ±1 п.п. при 95% уверенности?

<details><summary>▶️ Решение</summary>

$1.96\sqrt{0.25/n} \le 0.01 \Rightarrow n \ge 9604$. Ответ «около 10 000 бросков» и рассуждение через $\sigma/\sqrt{n}$ — то, что ждут.
</details>

### Задача 99. Почему $\sqrt{d}$, а не $d$
В attention делят скоры на $\sqrt{d}$. Что сломается, если делить на $d$?

<details><summary>▶️ Решение</summary>

Дисперсия $\mathbf{q}^\top\mathbf{k}$ равна $d$ (модуль 11); после деления на $\sqrt d$ — 1, на $d$ — $1/d$. При $d = 128$ скоры ≈ ±0.09, softmax почти равномерный: attention превращается в простое усреднение, модель с трудом фокусируется на нужных токенах, а градиенты по $\mathbf{q}, \mathbf{k}$ становятся очень малыми.

```python
d = 128; q, k = np.random.randn(d, 1000), np.random.randn(d, 1000)
s = (q * k).sum(0)
print(round(s.std(), 1), round((s / np.sqrt(d)).std(), 2), round((s / d).std(), 3))   # ≈ 11, 1.0, 0.09
```
</details>

### Задача 100. Где ошибается рассуждение
Кандидат говорит: «Наша модель увеличила средний чек на 5%, потому что в группе с моделью средний чек на 5% выше, чем в группе без неё. Группы — пользователи, которые сами включили рекомендации и которые не включили». Что не так и как исправить?

<details><summary>▶️ Решение</summary>

Самоотбор: включают рекомендации более вовлечённые пользователи — конфаундер (модуль 19). Нужно: рандомизированный A/B (включение по случайному флагу), или хотя бы propensity score / DiD с проверкой допущений; оценивать эффект с доверительным интервалом и следить за долгосрочными метриками (возвраты, удержание), а не только средним чеком.
</details>

---

### ✅ Чек-лист готовности

- [ ] Решил все 100 задач практикума и задачи «Практика модуля»
- [ ] Нашёл все 10 багов в задачах 61–70 до того, как открыл решения
- [ ] Реализовал с нуля хотя бы 5 мини-проектов из 81–90 и сверил с sklearn
- [ ] Прошёл [рабочую тетрадь с автопроверкой](exercises/README.md) — все тесты зелёные
- [ ] Могу без подсказок вывести градиенты логистической регрессии и softmax + CE
- [ ] Могу написать backprop двухслойной сети на NumPy за 15 минут
- [ ] Могу реализовать attention и объяснить каждую размерность
- [ ] Могу решить задачу про Байеса и про A/B-тест на бумаге
- [ ] Могу объяснить связь MLE ↔ лоссы, MAP ↔ регуляризация, CE ↔ KL

---

[← 23. 🧰 Прикладное: Фурье, свёртки и обработка сигналов](modules/23_fourier_signals.md) · [🏠 Оглавление](README.md#-оглавление) · [Шпаргалка на одной странице →](CHEATSHEET.md)
