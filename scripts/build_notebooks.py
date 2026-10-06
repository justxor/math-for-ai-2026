#!/usr/bin/env python3
"""Собирает Jupyter-ноутбуки по модулям из кода в README.md и BASICS.md.

Запуск из корня репозитория:  python scripts/build_notebooks.py
Ноутбуки появятся в notebooks/. Зависимостей, кроме стандартной библиотеки, нет.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
REPO = "justxor/math-for-ai-2026"

MODULES = {
    "m00b": "00b_basics_short", "m01": "01_notation", "m02": "02_linear_algebra", "m03": "03_calculus",
    "m04": "04_backprop", "m05": "05_optimization", "m06": "06_probability", "m07": "07_statistics",
    "m08": "08_information_theory", "m09": "09_losses_metrics", "m10": "10_classic_ml", "m11": "11_deep_learning",
    "m12": "12_generative", "m13": "13_numerics", "m14": "14_reinforcement_learning", "m15": "15_graphs_gnn",
    "m16": "16_learning_theory", "m17": "17_kernels_gp_ot", "m18": "18_time_series", "m19": "19_causal_inference",
    "m20": "20_retrieval_rag", "m21": "21_recommender_systems", "m22": "22_bayes_uncertainty", "m23": "23_fourier_signals",
    "practice": "practicum",
}


FIGURES = {
    "m00b": ["functions_gallery", "derivative_tangent"], "m01": ["softmax_temp"],
    "m02": ["dot_product", "matrix_transform", "eigen", "pca"], "m03": ["taylor"], "m04": ["activations"],
    "m05": ["gd_contours", "optimizers", "lr_schedule", "l1_l2", "convexity_saddle"],
    "m06": ["distributions", "mvn", "clt"], "m07": ["bias_variance"], "m08": ["entropy_kl", "kl_fit"],
    "m09": ["roc_pr", "calibration"], "m11": ["attention"], "m12": ["diffusion"], "m14": ["value_iteration"],
    "m15": ["spectral_clustering"], "m16": ["double_descent"], "m17": ["kernel_trick", "gaussian_process", "wasserstein"],
    "m18": ["time_series", "ts_cv"], "m19": ["simpson"], "m20": ["bm25"], "m21": ["matrix_factorization"],
    "m22": ["bayes_conformal"], "m23": ["fourier"],
    "basics": ["unit_circle", "integral_area", "equations", "sequences_limits", "descriptive_stats", "venn"],
}

VIZ_SETUP = """# Помощники для иллюстраций (те же, что в scripts/make_figures.py)
import matplotlib.pyplot as plt
from scipy import stats
C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

def save(fig, name):          # в ноутбуке просто показываем картинку
    fig.tight_layout(); plt.show()

def arrow(ax, v, color, label=None, origin=(0, 0), **kw):
    ax.annotate("", xy=(origin[0] + v[0], origin[1] + v[1]), xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, **kw))
    if label:
        ax.text(origin[0] + v[0] * 1.08, origin[1] + v[1] * 1.08, label, color=color, fontsize=13, fontweight="bold")

def rosen_like(x, y):
    return 0.5 * x**2 + 4 * y**2"""


def figure_sources():
    import ast
    src = (ROOT / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def colab(path):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{path})")


def section_cells(text, source_file, anchor):
    """Превращает фрагмент markdown в ячейки: контекст (заголовок/условие) + код."""
    cells, heading, task = [], "", ""
    in_details = False
    pos = 0
    for m in re.finditer(r"```python\n(.*?)```", text, re.S):
        before = text[pos:m.start()]
        for line in before.splitlines():
            s = line.strip()
            if re.match(r"^#{2,3} ", s):
                heading, task = s.lstrip("#").strip(), ""
            elif re.match(r"^\*\*(Б?\d+\.\d+\.)\*\*", s) or re.match(r"^### Задача \d+", s):
                task = s
                if s.startswith("### "):
                    heading, task = s[4:], ""
            if "<details>" in s:
                in_details = True
            if "</details>" in s:
                in_details = False
        ctx = []
        if heading:
            ctx.append(f"### {heading}")
        if task:
            ctx.append(task)
        if in_details:
            ctx.append("*▶️ Решение — сначала попробуйте сами.*")
        if ctx:
            cells.append(md("\n\n".join(ctx)))
        cells.append(code(m.group(1)))
        pos = m.end()
    return cells


def build(title, anchor, body, source_file, out_name, figs=(), fig_src=None):
    path = f"notebooks/{out_name}.ipynb"
    head = (f"# {title}\n\n{colab(path)}\n\n"
            f"Код из раздела [{source_file}](https://github.com/{REPO}/blob/main/{source_file}#{anchor}). "
            "Запускайте ячейки сверху вниз: последующие используют переменные предыдущих. "
            "Теория, формулы и картинки — в тексте курса по ссылке.")
    setup = ("# Colab: всё нужное уже установлено. Локально: pip install -r requirements.txt\n"
             "import numpy as np\nimport math\nnp.set_printoptions(precision=4, suppress=True)")
    cells = [md(head), code(setup)] + section_cells(body, source_file, anchor)
    if figs:
        cells.append(md("## 🔬 Лаборатория: иллюстрации модуля\n\n"
                        "Это код картинок из курса. Меняйте параметры (learning rate, температуру, число точек, "
                        "ядро…) и перезапускайте ячейку — так интуиция появляется быстрее всего."))
        cells.append(code(VIZ_SETUP))
        for f in figs:
            cells.append(code(fig_src[f] + f"\n\n{f}()"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:03d}"
    (OUT / f"{out_name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return sum(c["cell_type"] == "code" for c in cells) - 1


def main():
    OUT.mkdir(exist_ok=True)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    parts = re.split(r'<a id="([^"]+)"></a>', readme)
    sections, current = {}, None
    for anchor, body in zip(parts[1::2], parts[2::2]):
        if anchor in MODULES:
            current = anchor
            sections[current] = body
        elif current and (anchor.startswith(current + "-") or (current == "practice" and anchor.startswith("p-"))):
            sections[current] += body                  # вложенные якоря: m00b-practice, p-la, p-calc, …
        else:
            current = None
    index = []
    fig_src = figure_sources()
    for anchor, name in MODULES.items():
        body = sections[anchor]
        title = re.search(r"^# (.+)$", body, re.M).group(1)
        n = build(title, anchor, body, "README.md", name, FIGURES.get(anchor, ()), fig_src)
        index.append((name, title, n))
    basics = (ROOT / "BASICS.md").read_text(encoding="utf-8")
    n = build("Математическая база с нуля (BASICS.md)", "b1", basics, "BASICS.md", "00a_basics_full",
              FIGURES["basics"], fig_src)
    index.insert(0, ("00a_basics_full", "Математическая база с нуля (BASICS.md)", n))

    lines = ["# 📓 Ноутбуки курса\n",
             "Каждый ноутбук — весь код модуля по порядку, с заголовками и условиями задач. "
             "Откройте в Colab кнопкой или локально: `pip install -r requirements.txt jupyter && jupyter lab`.\n",
             "Ноутбуки генерируются скриптом `python scripts/build_notebooks.py` — правьте текст курса, а не ноутбуки.\n",
             "| Ноутбук | Ячеек с кодом | Открыть |", "|---------|---------------|---------|"]
    for name, title, n in index:
        lines.append(f"| [{title}]({name}.ipynb) | {n} | {colab(f'notebooks/{name}.ipynb')} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"собрано ноутбуков: {len(index)}")


if __name__ == "__main__":
    main()
