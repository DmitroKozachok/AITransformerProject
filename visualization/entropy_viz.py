"""
Теплова карта ентропії уваги: шари × голови
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
from utils.plot_utils import save_or_show


class EntropyVisualizer:
    def __init__(self, config):
        self.cfg = config

    def _compute_entropy_matrix(self, attentions) -> np.ndarray:
        """
        Повертає матрицю (num_layers × num_heads) із середньою ентропією
        рядків матриці уваги для кожної голови кожного шару.
        """
        num_layers = len(attentions)
        num_heads  = attentions[0].shape[1]
        matrix = np.zeros((num_layers, num_heads))

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                head_attn = attentions[layer_idx][0][head_idx].numpy()
                row_entropies = [entropy(row + 1e-9) for row in head_attn]
                matrix[layer_idx, head_idx] = np.mean(row_entropies)

        return matrix

    def plot_entropy_heatmap(self, attentions):
        """Малює теплову карту ентропії."""
        matrix    = self._compute_entropy_matrix(attentions)
        num_heads = matrix.shape[1]

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            matrix,
            cmap=self.cfg.ENTROPY_CMAP,
            xticklabels=[f"H{i+1}" for i in range(num_heads)],
            yticklabels=[f"L{i+1}" for i in range(len(attentions))],
            annot=True,
            fmt=".2f",
        )
        plt.title(
            "Ентропія уваги: шари × голови\n(вища = розсіяна, нижча = сфокусована)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.xlabel("Голова")
        plt.ylabel("Шар")
        plt.tight_layout()
        plt.show()
