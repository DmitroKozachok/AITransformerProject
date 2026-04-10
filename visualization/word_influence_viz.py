"""
Вплив конкретного токена на всі інші.

Для обраного токена будує "карту впливу" — як сильно він
притягує увагу інших токенів на кожному шарі.
Відповідає на питання: "Наскільки важливе слово X для моделі?"
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show


class WordInfluenceVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_word_influence(self, tokens, attentions, token_idx: int):
        """
        Два графіки:
        1. Heatmap (шари × токени): скільки уваги отримує token_idx від кожного токена на кожному шарі
        2. Лінійний графік: сумарний вплив по шарах
        """
        num_layers = len(attentions)
        seq_len    = len(tokens)

        # influence_matrix[layer, from_token] = увага from_token → token_idx
        influence_matrix = np.zeros((num_layers, seq_len))
        for layer_idx in range(num_layers):
            avg = attentions[layer_idx][0].mean(dim=0).numpy()
            # Стовпець token_idx = скільки уваги отримує цей токен від кожного рядка
            influence_matrix[layer_idx] = avg[:, token_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Heatmap ---
        import seaborn as sns
        sns.heatmap(
            influence_matrix,
            xticklabels=tokens,
            yticklabels=[f"L{i+1}" for i in range(num_layers)],
            cmap=self.cfg.CMAP,
            ax=ax1,
        )
        ax1.set_title(
            f"Хто і скільки уваги приділяє токену '{tokens[token_idx]}'",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax1.set_xlabel("Токен-джерело")
        ax1.set_ylabel("Шар")
        ax1.tick_params(axis="x", rotation=90)

        # --- Лінійний: сумарний вплив по шарах ---
        total_per_layer = influence_matrix.sum(axis=1)
        ax2.plot(
            range(1, num_layers + 1),
            total_per_layer,
            marker="o",
            linewidth=2,
            color="tomato",
        )
        ax2.fill_between(range(1, num_layers + 1), total_per_layer, alpha=0.15, color="tomato")
        ax2.set_xticks(range(1, num_layers + 1))
        ax2.set_xlabel("Шар")
        ax2.set_ylabel("Сумарна увага до токена")
        ax2.set_title(
            f"Сумарний вплив на '{tokens[token_idx]}' по шарах",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Аналіз впливу токена '{tokens[token_idx]}'",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, f"11_word_influence_{tokens[token_idx].replace('·','')}.png")
