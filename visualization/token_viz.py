"""
Еволюція уваги одного токена по всіх шарах
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class TokenVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_token_evolution(self, tokens, attentions, token_idx: int):
        """
        Стовпчикові графіки: як конкретний токен розподіляє увагу
        на кожному шарі.
        """
        num_layers = len(attentions)
        cols = self.cfg.GRID_COLS
        rows = math.ceil(num_layers / cols)

        cmap   = cm.get_cmap(self.cfg.CMAP)
        colors = [cmap(i / len(tokens)) for i in range(len(tokens))]

        fig, axes = plt.subplots(rows, cols, figsize=self.cfg.FIGURE_SIZE)
        axes = axes.flatten()

        for layer_idx in range(num_layers):
            avg_attn       = attentions[layer_idx][0].mean(dim=0).numpy()
            token_attention = avg_attn[token_idx]

            axes[layer_idx].bar(
                range(len(tokens)),
                token_attention,
                color=colors,
            )
            axes[layer_idx].set_xticks(range(len(tokens)))
            axes[layer_idx].set_xticklabels(tokens, rotation=90, fontsize=8)
            axes[layer_idx].set_title(f"Шар {layer_idx + 1}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            axes[layer_idx].set_ylim(0, 1)
            axes[layer_idx].set_ylabel("Увага")

        for j in range(num_layers, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Еволюція уваги токена '{tokens[token_idx]}' по шарах",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()