"""
Similarity matrix між шарами GPT-2.
Показує наскільки схожі патерни уваги між різними шарами —
дозволяє знайти надлишкові шари і точки де відбувається зміна представлення.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show


def _layer_avg_flat(attentions, layer_idx: int) -> np.ndarray:
    """Повертає плаский вектор усередненої матриці уваги шару."""
    avg = attentions[layer_idx][0].mean(dim=0).numpy()
    return avg.flatten()


class SimilarityVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_layer_similarity(self, attentions):
        """
        Будує матрицю косинусної схожості між усіма парами шарів.
        """
        num_layers = len(attentions)
        vecs = np.stack([_layer_avg_flat(attentions, i) for i in range(num_layers)])

        # Косинусна схожість
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / (norms + 1e-9)
        sim_matrix = normed @ normed.T

        labels = [f"L{i+1}" for i in range(num_layers)]

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            sim_matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap=self.cfg.SIMILARITY_CMAP,
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            linewidths=0.4,
            ax=ax,
        )
        ax.set_title(
            "Косинусна схожість між шарами\n"
            "(1.0 = ідентичні патерни, 0.0 = повністю різні)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        save_or_show(fig, "06_layer_similarity.png")
