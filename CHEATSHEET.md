[← Практикум: 100 задач с решениями](PRACTICE.md) · [🏠 Оглавление](README.md#-оглавление)

<a id="cheatsheet"></a>

# ⚡ Шпаргалка на одной странице

> Повторить за час до собеседования.

## Линейная алгебра
- `(m×k)·(k×n) → (m×n)`, стоимость ≈ 2mkn FLOP. $(AB)^\top = B^\top A^\top$.
- $\mathbf{a}^\top\mathbf{b} = \lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert\cos\theta$; косинус = скалярное произведение нормированных.
- $A\mathbf{v} = \lambda\mathbf{v}$; симметричная: $A = Q\Lambda Q^\top$; $\mathrm{tr} = \sum\lambda$, $\det = \prod\lambda$.
- SVD $A = U\Sigma V^\top$ для любой матрицы; лучшее приближение ранга k — top-k сингулярных.
- PCA = центрировать → SVD → проекция на первые k строк $V^\top$; доля дисперсии $\sigma_i^2/\sum\sigma_j^2$.
- LoRA: $\Delta W = BA$, ранг ≤ r, параметров $r(d+k)$ вместо $dk$.

## Анализ
- $\sigma' = \sigma(1-\sigma)$, $\tanh' = 1-\tanh^2$, $(\log x)' = 1/x$.
- $\nabla_\mathbf{x}\,\mathbf{a}^\top\mathbf{x} = \mathbf{a}$, $\nabla\,\mathbf{x}^\top A\mathbf{x} = (A + A^\top)\mathbf{x}$, $\nabla\lVert A\mathbf{x} - \mathbf{b}\rVert^2 = 2A^\top(A\mathbf{x} - \mathbf{b})$.
- Линейный слой $Y = XW$: $\partial L/\partial W = X^\top G$, $\partial L/\partial X = GW^\top$.
- **softmax + CE: $\nabla_\mathbf{z} = \mathbf{p} - \mathbf{y}$**; сигмоида + BCE: $\hat{y} - y$.
- Backprop: сложение раздаёт, умножение меняет местами, max маршрутизирует, ветвление суммирует.

## Оптимизация
- GD: $\theta \leftarrow \theta - \eta\nabla L$; устойчив при $\eta < 2/\lambda_{\max}$; шагов ∝ κ.
- Momentum: $\mathbf{v} \leftarrow \beta\mathbf{v} + \mathbf{g}$. Adam: EMA $\mathbf{g}$ и $\mathbf{g}^2$ + коррекция $1/(1-\beta^t)$.
- AdamW: weight decay вне адаптивного масштаба. Типично $\beta = (0.9, 0.95\text{–}0.999)$, wd 0.01–0.1.
- Warmup + cosine/WSD. Clipping по глобальной норме 1.0.
- L2 = гауссовский априор, L1 = лапласовский → разреженность.

## Вероятности и статистика
- Байес: $p(\theta\mid D) \propto p(D\mid\theta)p(\theta)$. Тест на болезнь: ~17%, не 99%.
- $\mathrm{Var}[aX+bY] = a^2\mathrm{Var}X + b^2\mathrm{Var}Y + 2ab\mathrm{Cov}$; $\mathrm{Var}[\bar X] = \sigma^2/n$.
- He-init: $\mathrm{Var}[w] = 2/n_{\text{in}}$; Xavier: $2/(n_{\text{in}} + n_{\text{out}})$.
- Репараметризация: $z = \mu + \sigma\varepsilon$.
- MLE = min NLL: гаусс → MSE, Бернулли → BCE, категориальное → CE, Лаплас → MAE.
- 95% ДИ: $\bar{x} \pm 1.96\,s/\sqrt{n}$. A/B: $n \approx 16p(1-p)/\delta^2$ на группу.
- p-value — $P(\text{данные} \mid H_0)$, не $P(H_0 \mid \text{данные})$. Не подглядывать.

## Теория информации
- $H(P) = -\sum p\log p$; $H(P,Q) = -\sum p\log q$; $D_{\mathrm{KL}}(P\parallel Q) = H(P,Q) - H(P) \ge 0$, несимметрична.
- Forward KL — mass-covering (MLE), reverse KL — mode-seeking (VI, RLHF).
- $D_{\mathrm{KL}}(\mathcal{N}(\mu,\sigma^2)\parallel \mathcal{N}(0,1)) = \frac{1}{2}(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$.
- Perplexity $= e^{\text{CE}}$.

## Метрики
- Precision $= TP/(TP+FP)$, Recall $= TP/(TP+FN)$, $F_1$ — гармоническое среднее.
- ROC-AUC = P(скор позитива > скор негатива). При дисбалансе смотреть PR-AUC.
- MSE → среднее, MAE → медиана, pinball → квантиль.

## Deep Learning
- Attention: $\mathrm{softmax}(QK^\top/\sqrt{d_k} + M)V$; $/\sqrt{d_k}$ — чтобы дисперсия скоров была 1.
- Self-attention $O(T^2d)$; KV-кэш $2LTn_{kv}d_h \cdot$ байт.
- Параметры трансформера $\approx 12Ld^2$; обучение ≈ $6ND$ FLOP; Chinchilla $D \approx 20N$.
- Conv: $\lfloor(H + 2P - K)/S\rfloor + 1$; параметры $C_{\text{out}}(C_{\text{in}}K^2 + 1)$.
- LayerNorm по признакам, BatchNorm по батчу; Pre-LN стабильнее.
- Память обучения с AdamW в mixed precision ≈ 16 байт/параметр + активации.

## Генеративные
- ELBO = реконструкция − KL(q(z|x) ‖ p(z)).
- Диффузия: $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\varepsilon$, лосс $\lVert\boldsymbol\varepsilon - \boldsymbol\varepsilon_\theta\rVert^2$; CFG: $\boldsymbol\varepsilon_\varnothing + w(\boldsymbol\varepsilon_c - \boldsymbol\varepsilon_\varnothing)$.
- DPO: $-\log\sigma\bigl(\beta[\log\frac{\pi}{\pi_{\text{ref}}}(y_w) - \log\frac{\pi}{\pi_{\text{ref}}}(y_l)]\bigr)$.

## 🚀 Продвинутое
- RL: $V(s) = \mathbb{E}[r + \gamma V(s')]$; $A = Q - V$; policy gradient $\mathbb{E}[\nabla\log\pi \cdot A]$; PPO — clip $\pi/\pi_{\text{old}}$ в $[1-\epsilon, 1+\epsilon]$; GRPO — advantage по группе ответов.
- Графы: $L = D - A$, $\mathbf{x}^\top L\mathbf{x} = \sum (x_i - x_j)^2$; нулей в спектре = компонент; GCN $= \sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$.
- Обобщение: Хёфдинг $2e^{-2N\varepsilon^2}$; $d_{VC}$ линейного классификатора $= d + 1$; GD из нуля → решение минимальной нормы; double descent.
- Ядра: матрица Грама ⪰ 0; GP: $\mu_* = K_{*X}(K + \sigma^2I)^{-1}\mathbf{y}$; $W_1$ — стоимость перевозки массы, FID — $W_2$ между гауссианами.

## 🧰 Прикладное
- Ряды: $y = T + S + R$; стационарность — разности/лог; AR(1) стационарен при $\lvert\varphi\rvert < 1$; валидация только по времени; бейзлайн — сезонный наивный; WAPE, MASE.
- Причинность: ATE $= \mathbb{E}[Y(1) - Y(0)]$; рандомизация убирает смещение отбора; контролировать конфаундеры, не коллайдеры; DiD, IPW $T/e + (1-T)/(1-e)$; uplift.
- Поиск: BM25 (насыщение $k_1$, длина $b$, IDF); после нормировки $\lVert a - b\rVert^2 = 2 - 2\cos$; IVF-PQ, HNSW; RRF $\sum 1/(k + \text{rank})$; cross-encoder для реранкинга.
- Рекомендации: $\hat r_{ui} = \mathbf{p}_u^\top\mathbf{q}_i + b_u + b_i + \mu$, ALS; two-tower + ANN; NDCG $= \text{DCG}/\text{IDCG}$; logQ-коррекция.
- Неопределённость: алеаторная vs эпистемическая; Beta–Бернулли; ансамбли, MC-dropout; conformal: квантиль ошибок калибровки → покрытие ≥ $1 - \alpha$.
- Фурье: FFT $O(N\log N)$; свёртка ↔ умножение спектров; Найквист $f_s/2$; STFT → мел → log.

## Численная устойчивость
- softmax и logsumexp — через вычитание максимума; лоссы — из логитов.
- BF16 — диапазон FP32, мало мантиссы; мастер-веса в FP32. FP16 — нужен loss scaling.
- NaN: найти первый шаг → данные → лосс (log 0, sqrt 0) → норма градиента → LR/clipping/BF16.

---

[← Практикум: 100 задач с решениями](PRACTICE.md) · [🏠 Оглавление](README.md#-оглавление)
