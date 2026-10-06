#!/usr/bin/env python3
"""Проверяет, что весь Python-код курса запускается.

Код из BASICS.md, README.md, modules/*.md, PRACTICE.md и CHEATSHEET.md группируется по разделам (якорям <a id="...">) и выполняется
по порядку в отдельном процессе для каждого раздела — так же, как читатель запускал бы его подряд.
Блоки с PyTorch пропускаются, если torch не установлен.

Запуск:  python tests/check_snippets.py
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRELUDE = "import numpy as np, math\nimport warnings; warnings.filterwarnings('ignore')\n"
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def sections(path):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r'<a id="([^"]+)"></a>', text)
    yield "intro", parts[0]
    yield from zip(parts[1::2], parts[2::2])


def main():
    failed, total, skipped = [], 0, 0
    files = ["BASICS.md", "README.md"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "modules").glob("*.md")) \
        + ["PRACTICE.md", "CHEATSHEET.md"]
    for name in files:
        for anchor, body in sections(ROOT / name):
            blocks = re.findall(r"```python\n(.*?)```", body, re.S)
            if not HAS_TORCH:
                skipped += sum("torch" in b for b in blocks)
                blocks = [b for b in blocks if "torch" not in b]
            if not blocks:
                continue
            total += 1
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(PRELUDE + "\n".join(blocks))
            r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=600,
                               env={**os.environ, "MPLBACKEND": "Agg"})
            status = "ok " if r.returncode == 0 else "FAIL"
            print(f"{status} {name}#{anchor} ({len(blocks)} блоков)")
            if r.returncode != 0:
                failed.append(f"{name}#{anchor}")
                print(r.stderr[-1500:])
    print(f"\nразделов проверено: {total}, с ошибками: {len(failed)}, пропущено torch-блоков: {skipped}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
