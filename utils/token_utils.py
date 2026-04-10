"""
Утиліти для роботи з токенами GPT-2
"""


def clean_tokens(tokens) -> list[str]:
    """
    GPT-2 позначає пробіл перед словом символом Ġ (U+0120).
    Замінюємо:
      Ġword  →  ·word   (пробіл перед словом)
      Ċ      →  \\n     (новий рядок)
    """
    result = []
    for t in tokens:
        t = t.replace("\u0120", "·")   # Ġ → ·
        t = t.replace("\u010a", "\\n") # Ċ → \n
        result.append(t)
    return result
