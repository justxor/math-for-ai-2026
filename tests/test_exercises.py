#!/usr/bin/env python3
"""Прогоняет все автопроверки рабочей тетради на эталонных решениях.

Запуск:  python tests/test_exercises.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exercises"))
import checks  # noqa: E402
import solutions as s  # noqa: E402

PAIRS = [
    (checks.check_sigmoid, s.sigmoid), (checks.check_softmax, s.softmax),
    (checks.check_cross_entropy, s.cross_entropy), (checks.check_cosine_matrix, s.cosine_matrix),
    (checks.check_pca, s.pca), (checks.check_linear_regression, s.linear_regression),
    (checks.check_gradient_descent, s.gradient_descent), (checks.check_numerical_gradient, s.numerical_gradient),
    (checks.check_adam_step, s.adam_step), (checks.check_bayes, s.bayes_posterior),
    (checks.check_bootstrap, s.bootstrap_ci), (checks.check_entropy, s.entropy_bits),
    (checks.check_kl, s.kl_divergence), (checks.check_prf, s.precision_recall_f1),
    (checks.check_auc, s.roc_auc), (checks.check_kmeans_step, s.kmeans_step),
    (checks.check_attention, s.attention), (checks.check_conv, s.conv_output_size),
    (checks.check_ndcg, s.ndcg_at_k), (checks.check_conformal, s.conformal_quantile),
]

if __name__ == "__main__":
    for check, fn in PAIRS:
        check(fn)
    print(f"\nвсе {len(PAIRS)} упражнений: эталонные решения проходят проверки")
