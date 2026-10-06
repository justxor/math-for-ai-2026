# 📖 Глоссарий: 162 терминов математики для ИИ — русский ↔ английский

> Статьи, документация и библиотеки — на английском, а учиться удобнее на русском. Здесь каждый термин курса с переводом, определением в одну строку и ссылкой на модуль, где он разобран.
> Ищите через **Ctrl+F** — по-русски или по-английски.

Связанные материалы: [основной курс](README.md) · [математика с нуля](BASICS.md) · [ноутбуки](notebooks/README.md) · [колода Anki](anki/README.md)

## Разделы

- [📐 Линейная алгебра](#g0) — 19
- [📈 Анализ и оптимизация](#g1) — 22
- [🎲 Вероятность и статистика](#g2) — 27
- [🔤 Теория информации, лоссы, метрики](#g3) — 14
- [🧠 Модели и глубокое обучение](#g4) — 22
- [✨ Генеративные модели и RL](#g5) — 17
- [🚀 Продвинутое](#g6) — 11
- [🧰 Прикладное](#g7) — 24
- [🔢 Числа и вычисления](#g8) — 6

<a id="g0"></a>

## 📐 Линейная алгебра

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Вектор** | *vector* | упорядоченный список чисел; точка или стрелка в пространстве | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Матрица** | *matrix* | таблица чисел; линейное преобразование; слой нейросети | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Тензор** | *tensor* | многомерный массив, например (batch, seq, dim) | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Скалярное произведение** | *dot product, inner product* | сумма покомпонентных произведений; мера сонаправленности | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Косинусное сходство** | *cosine similarity* | скалярное произведение нормированных векторов, от −1 до 1 | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Норма** | *norm* | «длина» вектора: L1, L2, L∞ | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Транспонирование** | *transpose* | строки ↔ столбцы | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Ранг** | *rank* | число линейно независимых столбцов | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Определитель** | *determinant* | во сколько раз преобразование меняет объём | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Обратная матрица** | *inverse matrix* | A⁻¹: AA⁻¹ = I | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Собственный вектор / число** | *eigenvector / eigenvalue* | Av = λv — направление, которое только растягивается | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Сингулярное разложение** | *singular value decomposition, SVD* | A = UΣVᵀ для любой матрицы | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Метод главных компонент** | *principal component analysis, PCA* | проекция на направления наибольшей дисперсии | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Ортогональная матрица** | *orthogonal matrix* | QᵀQ = I, сохраняет длины (повороты) | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Положительно определённая** | *positive definite* | xᵀAx > 0 для всех x ≠ 0 | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Число обусловленности** | *condition number* | σmax/σmin; чувствительность к ошибкам, скорость GD | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Низкоранговое приближение** | *low-rank approximation* | A ≈ UVᵀ малого ранга; основа LoRA | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Трансляция размерностей** | *broadcasting* | автоматическое растягивание осей при операциях | [модуль 02](modules/02_linear_algebra.md#m02) |
| **Псевдообратная** | *pseudoinverse, Moore–Penrose* | X⁺; решение минимальной нормы | [модуль 16](modules/16_learning_theory.md#m16) |

<a id="g1"></a>

## 📈 Анализ и оптимизация

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Производная** | *derivative* | скорость изменения функции, наклон касательной | [модуль 03](modules/03_calculus.md#m03) |
| **Частная производная** | *partial derivative* | производная по одной переменной при фиксированных остальных | [модуль 03](modules/03_calculus.md#m03) |
| **Градиент** | *gradient* | вектор частных производных, направление наискорейшего роста | [модуль 03](modules/03_calculus.md#m03) |
| **Якобиан** | *Jacobian* | матрица первых производных векторной функции | [модуль 03](modules/03_calculus.md#m03) |
| **Гессиан** | *Hessian* | матрица вторых производных, кривизна | [модуль 03](modules/03_calculus.md#m03) |
| **Цепное правило** | *chain rule* | производная композиции = произведение производных | [модуль 03](modules/03_calculus.md#m03) |
| **Ряд Тейлора** | *Taylor series* | приближение функции многочленом около точки | [модуль 03](modules/03_calculus.md#m03) |
| **Обратное распространение ошибки** | *backpropagation* | цепное правило по графу вычислений от лосса к весам | [модуль 04](modules/04_backprop.md#m04) |
| **Автодифференцирование** | *automatic differentiation, autograd* | автоматический точный расчёт производных программы | [модуль 04](modules/04_backprop.md#m04) |
| **Граф вычислений** | *computational graph* | операции как узлы, данные как рёбра | [модуль 04](modules/04_backprop.md#m04) |
| **Затухающие / взрывающиеся градиенты** | *vanishing / exploding gradients* | градиент → 0 или ∞ через много слоёв | [модуль 04](modules/04_backprop.md#m04) |
| **Градиентный спуск** | *gradient descent* | θ ← θ − η∇L | [модуль 05](modules/05_optimization.md#m05) |
| **Стохастический градиентный спуск** | *stochastic gradient descent, SGD* | градиент по мини-батчу | [модуль 05](modules/05_optimization.md#m05) |
| **Скорость обучения** | *learning rate, step size* | размер шага η | [модуль 05](modules/05_optimization.md#m05) |
| **Импульс** | *momentum* | накопление скорости по прошлым градиентам | [модуль 05](modules/05_optimization.md#m05) |
| **Расписание LR** | *learning rate schedule* | warmup, cosine decay, WSD | [модуль 05](modules/05_optimization.md#m05) |
| **Выпуклая функция** | *convex function* | локальный минимум = глобальный | [модуль 05](modules/05_optimization.md#m05) |
| **Седловая точка** | *saddle point* | градиент 0, но не минимум и не максимум | [модуль 05](modules/05_optimization.md#m05) |
| **Регуляризация** | *regularization* | штраф за сложность: L1, L2, dropout | [модуль 05](modules/05_optimization.md#m05) |
| **Затухание весов** | *weight decay* | сжатие весов к нулю на каждом шаге | [модуль 05](modules/05_optimization.md#m05) |
| **Множители Лагранжа** | *Lagrange multipliers* | оптимизация с ограничениями | [модуль 05](modules/05_optimization.md#m05) |
| **Обрезка градиента** | *gradient clipping* | ограничение нормы градиента | [модуль 05](modules/05_optimization.md#m05) |

<a id="g2"></a>

## 🎲 Вероятность и статистика

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Случайная величина** | *random variable* | величина, значение которой определяется случайно | [модуль 06](modules/06_probability.md#m06) |
| **Распределение** | *distribution* | как вероятность распределена по значениям | [модуль 06](modules/06_probability.md#m06) |
| **Плотность вероятности** | *probability density function, PDF* | вероятность = площадь под ней | [модуль 06](modules/06_probability.md#m06) |
| **Функция распределения** | *cumulative distribution function, CDF* | P(X ≤ x) | [модуль 06](modules/06_probability.md#m06) |
| **Матожидание** | *expected value, expectation* | среднее по распределению | [модуль 06](modules/06_probability.md#m06) |
| **Дисперсия** | *variance* | средний квадрат отклонения от среднего | [модуль 06](modules/06_probability.md#m06) |
| **Ковариация** | *covariance* | совместная изменчивость двух величин | [модуль 06](modules/06_probability.md#m06) |
| **Условная вероятность** | *conditional probability* | P(A \| B) | [модуль 06](modules/06_probability.md#m06) |
| **Формула Байеса** | *Bayes' rule* | апостериор ∝ правдоподобие × априор | [модуль 06](modules/06_probability.md#m06) |
| **Априорное / апостериорное** | *prior / posterior* | до / после наблюдения данных | [модуль 06](modules/06_probability.md#m06) |
| **Правдоподобие** | *likelihood* | вероятность данных при параметрах | [модуль 07](modules/07_statistics.md#m07) |
| **Независимые одинаково распределённые** | *i.i.d.* | основное допущение большинства методов ML | [модуль 06](modules/06_probability.md#m06) |
| **Закон больших чисел** | *law of large numbers* | выборочное среднее → матожидание | [модуль 06](modules/06_probability.md#m06) |
| **Центральная предельная теорема** | *central limit theorem, CLT* | сумма многих величин ≈ нормальная | [модуль 06](modules/06_probability.md#m06) |
| **Нормальное распределение** | *normal / Gaussian distribution* | колокол N(μ, σ²) | [модуль 06](modules/06_probability.md#m06) |
| **Трюк репараметризации** | *reparameterization trick* | z = μ + σε — градиент через сэмплирование | [модуль 06](modules/06_probability.md#m06) |
| **Метод максимального правдоподобия** | *maximum likelihood estimation, MLE* | параметры, максимизирующие правдоподобие | [модуль 07](modules/07_statistics.md#m07) |
| **Апостериорный максимум** | *maximum a posteriori, MAP* | MLE + априор = регуляризация | [модуль 07](modules/07_statistics.md#m07) |
| **Смещение и разброс** | *bias–variance tradeoff* | ошибка = bias² + variance + шум | [модуль 07](modules/07_statistics.md#m07) |
| **Доверительный интервал** | *confidence interval* | диапазон, накрывающий параметр в 95% повторов | [модуль 07](modules/07_statistics.md#m07) |
| **Проверка гипотез** | *hypothesis testing* | H₀, статистика, p-value | [модуль 07](modules/07_statistics.md#m07) |
| **p-значение** | *p-value* | P(данные не менее экстремальны \| H₀) | [модуль 07](modules/07_statistics.md#m07) |
| **Ошибка I / II рода** | *type I / II error* | ложная тревога / пропуск эффекта | [модуль 07](modules/07_statistics.md#m07) |
| **Мощность теста** | *statistical power* | 1 − β, вероятность найти реальный эффект | [модуль 07](modules/07_statistics.md#m07) |
| **Бутстрап** | *bootstrap* | пересэмплирование с возвращением | [модуль 07](modules/07_statistics.md#m07) |
| **Множественные сравнения** | *multiple comparisons* | Бонферрони, FDR | [модуль 07](modules/07_statistics.md#m07) |
| **A/B-тест** | *A/B test, randomized controlled experiment* | рандомизированное сравнение вариантов | [модуль 07](modules/07_statistics.md#m07) |

<a id="g3"></a>

## 🔤 Теория информации, лоссы, метрики

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Энтропия** | *entropy* | средняя неожиданность, мера неопределённости | [модуль 08](modules/08_information_theory.md#m08) |
| **Кросс-энтропия** | *cross-entropy* | основной лосс классификации и LLM | [модуль 08](modules/08_information_theory.md#m08) |
| **Дивергенция Кульбака–Лейблера** | *Kullback–Leibler divergence, KL* | «расстояние» между распределениями, несимметрично | [модуль 08](modules/08_information_theory.md#m08) |
| **Взаимная информация** | *mutual information* | насколько одна величина говорит о другой | [модуль 08](modules/08_information_theory.md#m08) |
| **Перплексия** | *perplexity* | exp(CE) — эффективное число вариантов | [модуль 08](modules/08_information_theory.md#m08) |
| **Функция потерь** | *loss function* | что минимизирует модель | [модуль 09](modules/09_losses_metrics.md#m09) |
| **Логит** | *logit* | выход модели до softmax/сигмоиды | [модуль 01](modules/01_notation.md#m01) |
| **Сглаживание меток** | *label smoothing* | мягкие целевые вероятности | [модуль 09](modules/09_losses_metrics.md#m09) |
| **Матрица ошибок** | *confusion matrix* | TP, FP, FN, TN | [модуль 09](modules/09_losses_metrics.md#m09) |
| **Точность / полнота** | *precision / recall* | доля верных среди найденных / найденных среди верных | [модуль 09](modules/09_losses_metrics.md#m09) |
| **F1-мера** | *F1 score* | гармоническое среднее precision и recall | [модуль 09](modules/09_losses_metrics.md#m09) |
| **ROC-AUC** | *area under ROC curve* | P(скор позитива > скор негатива) | [модуль 09](modules/09_losses_metrics.md#m09) |
| **Калибровка** | *calibration* | предсказанные вероятности = реальные частоты | [модуль 09](modules/09_losses_metrics.md#m09) |
| **Контрастное обучение** | *contrastive learning* | сближать пары-позитивы, отдалять негативы | [модуль 09](modules/09_losses_metrics.md#m09) |

<a id="g4"></a>

## 🧠 Модели и глубокое обучение

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Линейная регрессия** | *linear regression* | предсказание линейной функцией признаков | [модуль 10](modules/10_classic_ml.md#m10) |
| **Логистическая регрессия** | *logistic regression* | сигмоида от линейной функции | [модуль 10](modules/10_classic_ml.md#m10) |
| **Метод опорных векторов** | *support vector machine, SVM* | разделение с максимальным зазором | [модуль 10](modules/10_classic_ml.md#m10) |
| **Ядровой трюк** | *kernel trick* | скалярные произведения в пространстве признаков без его построения | [модуль 17](modules/17_kernels_gp_ot.md#m17) |
| **Проклятие размерности** | *curse of dimensionality* | в высоких размерностях данные разрежены | [модуль 10](modules/10_classic_ml.md#m10) |
| **Дерево решений** | *decision tree* | последовательность пороговых разбиений | [модуль 10](modules/10_classic_ml.md#m10) |
| **Бэггинг / бустинг** | *bagging / boosting* | параллельный ансамбль / последовательный по остаткам | [модуль 10](modules/10_classic_ml.md#m10) |
| **Кластеризация** | *clustering* | группировка без меток: k-means, GMM | [модуль 10](modules/10_classic_ml.md#m10) |
| **Функция активации** | *activation function* | нелинейность: ReLU, GELU, SiLU | [модуль 11](modules/11_deep_learning.md#m11) |
| **Инициализация весов** | *weight initialization* | Xavier, He — сохранить масштаб сигнала | [модуль 11](modules/11_deep_learning.md#m11) |
| **Нормализация по батчу / слою** | *batch norm / layer norm* | нормировка активаций | [модуль 11](modules/11_deep_learning.md#m11) |
| **Остаточная связь** | *residual connection, skip connection* | x + f(x) | [модуль 11](modules/11_deep_learning.md#m11) |
| **Свёртка** | *convolution* | скользящее окно с общими весами | [модуль 11](modules/11_deep_learning.md#m11) |
| **Механизм внимания** | *attention* | softmax(QKᵀ/√d)V | [модуль 11](modules/11_deep_learning.md#m11) |
| **Многоголовое внимание** | *multi-head attention* | несколько параллельных attention | [модуль 11](modules/11_deep_learning.md#m11) |
| **Позиционное кодирование** | *positional encoding, RoPE* | информация о порядке токенов | [модуль 11](modules/11_deep_learning.md#m11) |
| **KV-кэш** | *KV cache* | сохранённые ключи и значения при генерации | [модуль 11](modules/11_deep_learning.md#m11) |
| **Смесь экспертов** | *mixture of experts, MoE* | роутер выбирает часть подсетей | [модуль 11](modules/11_deep_learning.md#m11) |
| **Низкоранговая адаптация** | *low-rank adaptation, LoRA* | дообучение через ΔW = BA | [модуль 11](modules/11_deep_learning.md#m11) |
| **Квантизация** | *quantization* | хранение весов в 8/4 битах | [модуль 11](modules/11_deep_learning.md#m11) |
| **Дистилляция** | *knowledge distillation* | ученик повторяет мягкие ответы учителя | [модуль 11](modules/11_deep_learning.md#m11) |
| **Законы масштабирования** | *scaling laws* | степенная зависимость лосса от N, D, C | [модуль 16](modules/16_learning_theory.md#m16) |

<a id="g5"></a>

## ✨ Генеративные модели и RL

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Авторегрессионная модель** | *autoregressive model* | p(x) = ∏ p(xₜ \| x<ₜ) | [модуль 12](modules/12_generative.md#m12) |
| **Температура / top-k / top-p** | *temperature / top-k / nucleus sampling* | стратегии сэмплирования токенов | [модуль 12](modules/12_generative.md#m12) |
| **Вариационный автоэнкодер** | *variational autoencoder, VAE* | латентная модель, обучение через ELBO | [модуль 12](modules/12_generative.md#m12) |
| **Нижняя вариационная оценка** | *evidence lower bound, ELBO* | реконструкция − KL | [модуль 12](modules/12_generative.md#m12) |
| **Генеративно-состязательная сеть** | *generative adversarial network, GAN* | игра генератора и дискриминатора | [модуль 12](modules/12_generative.md#m12) |
| **Диффузионная модель** | *diffusion model* | учимся убирать шум | [модуль 12](modules/12_generative.md#m12) |
| **Бесклассификаторное наведение** | *classifier-free guidance, CFG* | усиление условия в диффузии | [модуль 12](modules/12_generative.md#m12) |
| **Обучение с подкреплением на отзывах людей** | *RLHF* | reward model + PPO с KL-штрафом | [модуль 12](modules/12_generative.md#m12) |
| **Прямая оптимизация предпочтений** | *direct preference optimization, DPO* | выравнивание без RL | [модуль 12](modules/12_generative.md#m12) |
| **Марковский процесс принятия решений** | *Markov decision process, MDP* | состояния, действия, награды, переходы | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Политика** | *policy* | π(a \| s) | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Функция ценности** | *value function* | V(s), Q(s, a) | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Преимущество** | *advantage* | A = Q − V | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Уравнение Беллмана** | *Bellman equation* | рекурсия для ценности | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Градиент политики** | *policy gradient* | E[∇log π · A] | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Исследование и использование** | *exploration vs exploitation* | пробовать новое или пользоваться лучшим | [модуль 14](modules/14_reinforcement_learning.md#m14) |
| **Многорукий бандит** | *multi-armed bandit* | RL без состояний | [модуль 14](modules/14_reinforcement_learning.md#m14) |

<a id="g6"></a>

## 🚀 Продвинутое

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Лапласиан графа** | *graph Laplacian* | L = D − A | [модуль 15](modules/15_graphs_gnn.md#m15) |
| **Графовая нейросеть** | *graph neural network, GNN* | передача сообщений между соседями | [модуль 15](modules/15_graphs_gnn.md#m15) |
| **Чрезмерное сглаживание** | *over-smoothing* | признаки вершин сливаются с глубиной | [модуль 15](modules/15_graphs_gnn.md#m15) |
| **VC-размерность** | *VC dimension* | ёмкость класса моделей | [модуль 16](modules/16_learning_theory.md#m16) |
| **Разрыв обобщения** | *generalization gap* | риск на тесте − риск на трейне | [модуль 16](modules/16_learning_theory.md#m16) |
| **Двойной спуск** | *double descent* | ошибка снова падает после порога интерполяции | [модуль 16](modules/16_learning_theory.md#m16) |
| **Неявная регуляризация** | *implicit regularization* | оптимизатор сам выбирает «простое» решение | [модуль 16](modules/16_learning_theory.md#m16) |
| **Гауссовский процесс** | *Gaussian process, GP* | распределение над функциями | [модуль 17](modules/17_kernels_gp_ot.md#m17) |
| **Байесовская оптимизация** | *Bayesian optimization* | подбор гиперпараметров с суррогатом | [модуль 17](modules/17_kernels_gp_ot.md#m17) |
| **Оптимальный транспорт** | *optimal transport* | минимальная стоимость перевозки массы | [модуль 17](modules/17_kernels_gp_ot.md#m17) |
| **Расстояние Вассерштейна** | *Wasserstein distance, earth mover's distance* | метрика OT | [модуль 17](modules/17_kernels_gp_ot.md#m17) |

<a id="g7"></a>

## 🧰 Прикладное

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Временной ряд** | *time series* | наблюдения, упорядоченные во времени | [модуль 18](modules/18_time_series.md#m18) |
| **Стационарность** | *stationarity* | статистики не меняются во времени | [модуль 18](modules/18_time_series.md#m18) |
| **Автокорреляция** | *autocorrelation, ACF* | корреляция ряда с собой со сдвигом | [модуль 18](modules/18_time_series.md#m18) |
| **Сезонность / тренд** | *seasonality / trend* | периодическая / долгосрочная компонента | [модуль 18](modules/18_time_series.md#m18) |
| **Горизонт прогноза** | *forecast horizon* | на сколько шагов вперёд прогноз | [модуль 18](modules/18_time_series.md#m18) |
| **Конфаундер** | *confounder* | общая причина воздействия и исхода | [модуль 19](modules/19_causal_inference.md#m19) |
| **Коллайдер** | *collider* | общее следствие; контроль по нему создаёт ложную связь | [модуль 19](modules/19_causal_inference.md#m19) |
| **Средний эффект воздействия** | *average treatment effect, ATE* | E[Y(1) − Y(0)] | [модуль 19](modules/19_causal_inference.md#m19) |
| **Склонность к воздействию** | *propensity score* | P(T = 1 \| x) | [модуль 19](modules/19_causal_inference.md#m19) |
| **Разность разностей** | *difference-in-differences, DiD* | сравнение изменений тест vs контроль | [модуль 19](modules/19_causal_inference.md#m19) |
| **Аплифт-моделирование** | *uplift modeling* | предсказание эффекта воздействия на объект | [модуль 19](modules/19_causal_inference.md#m19) |
| **Поиск с дополнением генерации** | *retrieval-augmented generation, RAG* | LLM отвечает по найденным документам | [модуль 20](modules/20_retrieval_rag.md#m20) |
| **Эмбеддинг** | *embedding* | векторное представление объекта | [модуль 20](modules/20_retrieval_rag.md#m20) |
| **Приближённый поиск соседей** | *approximate nearest neighbors, ANN* | HNSW, IVF, PQ | [модуль 20](modules/20_retrieval_rag.md#m20) |
| **Реранкер** | *reranker, cross-encoder* | точная переоценка короткого списка кандидатов | [модуль 20](modules/20_retrieval_rag.md#m20) |
| **Матричная факторизация** | *matrix factorization* | R ≈ PQᵀ | [модуль 21](modules/21_recommender_systems.md#m21) |
| **Холодный старт** | *cold start* | нет истории у нового пользователя или товара | [модуль 21](modules/21_recommender_systems.md#m21) |
| **Неявный отклик** | *implicit feedback* | клики и просмотры вместо оценок | [модуль 21](modules/21_recommender_systems.md#m21) |
| **Алеаторная / эпистемическая неопределённость** | *aleatoric / epistemic uncertainty* | шум данных / незнание модели | [модуль 22](modules/22_bayes_uncertainty.md#m22) |
| **Сопряжённый априор** | *conjugate prior* | апостериор того же семейства | [модуль 22](modules/22_bayes_uncertainty.md#m22) |
| **Конформное предсказание** | *conformal prediction* | интервалы с гарантией покрытия | [модуль 22](modules/22_bayes_uncertainty.md#m22) |
| **Преобразование Фурье** | *Fourier transform, FFT* | разложение сигнала на частоты | [модуль 23](modules/23_fourier_signals.md#m23) |
| **Спектрограмма** | *spectrogram, STFT* | спектр в скользящих окнах | [модуль 23](modules/23_fourier_signals.md#m23) |
| **Частота Найквиста** | *Nyquist frequency* | fs/2 — предел различимых частот | [модуль 23](modules/23_fourier_signals.md#m23) |

<a id="g8"></a>

## 🔢 Числа и вычисления

| Русский | English | Коротко | Где в курсе |
|---------|---------|---------|-------------|
| **Число с плавающей точкой** | *floating point* | FP32, FP16, BF16, FP8 | [модуль 13](modules/13_numerics.md#m13) |
| **Смешанная точность** | *mixed precision* | вычисления в BF16, мастер-веса в FP32 | [модуль 13](modules/13_numerics.md#m13) |
| **Масштабирование лосса** | *loss scaling* | защита малых градиентов в FP16 | [модуль 13](modules/13_numerics.md#m13) |
| **Логарифм суммы экспонент** | *log-sum-exp* | стабильный расчёт log Σ eˣ | [модуль 13](modules/13_numerics.md#m13) |
| **Переполнение / потеря порядка** | *overflow / underflow* | число слишком большое / маленькое для формата | [модуль 13](modules/13_numerics.md#m13) |
| **Утечка данных** | *data leakage* | информация из теста или будущего в обучении | [модуль 18](modules/18_time_series.md#m18) |

---

Не хватает термина? Откройте [Issue](https://github.com/justxor/math-for-ai-2026/issues) — добавим.
