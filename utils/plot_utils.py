"""
Утиліта для збереження/відображення графіків.
Всі фігури реєструються заздалегідь, а plt.show() викликається ОДИН РАЗ —
тоді всі вікна з'являються одночасно і між ними можна переключатися.
"""

import os
import matplotlib.pyplot as plt

_pending: list[tuple] = []
_output_dir: str = "results/latest/plots"


def set_output_dir(path: str):
    """Встановлює папку для збереження — викликати до перших save_or_show."""
    global _output_dir
    _output_dir = path
    os.makedirs(path, exist_ok=True)


def save_or_show(fig, filename: str, output_dir: str = None):
    """
    Зберігає фігуру у PNG і додає до черги.
    Вікно НЕ відкривається тут — лише після виклику wait_all().
    """
    folder = output_dir or _output_dir
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"  ✓ Збережено: {filepath}")

    title = filename.replace(".png", "").replace("_", " ")
    _pending.append((fig, title))


def wait_all():
    """
    Відкриває ВСІ вікна одночасно одним викликом plt.show().
    Блокує до закриття всіх вікон.
    """
    if not _pending:
        return

    for fig, title in _pending:
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

    print(f"\n── {len(_pending)} вікон відкрито — переключайся між ними. Закрий всі щоб завершити ──")
    plt.show()
    _pending.clear()