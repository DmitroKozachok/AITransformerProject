"""
Головні патерни — кластеризація голів уваги.

144 голови (12 шарів × 12 голів) кластеризуються за схожістю
їх поведінки. Виявляє спеціалізовані ролі:
  - syntactic heads  (зв'язок підмет-присудок)
  - previous-token heads (увага на попередній токен)
  - global heads (рівномірна увага)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.plot_utils import save_or_show


class HeadPatternsVisualizer:
    def __init__(self, config, n_clusters: int = 4):
        self.cfg        = config
        self.n_clusters = n_clusters

    def _extract_head_vectors(self, attentions) -> tuple[np.ndarray, list[str]]:
        """
        Повертає (matrix, labels) де кожен рядок — плаский вектор однієї голови.
        """
        vectors = []
        labels  = []
        for layer_idx, attn_tensor in enumerate(attentions):
            num_heads = attn_tensor.shape[1]
            for head_idx in range(num_heads):
                vec = attn_tensor[0][head_idx].numpy().flatten()
                vectors.append(vec)
                labels.append(f"L{layer_idx+1}H{head_idx+1}")
        return np.stack(vectors), labels

    def plot_head_clusters(self, attentions):
        """
        PCA → 2D + KMeans кластеризація всіх голів.
        Кожна точка = одна голова, колір = кластер.
        """
        vectors, labels = self._extract_head_vectors(attentions)

        # Зменшуємо розмірність до 2D
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors)

        # Кластеризація
        kmeans  = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster = kmeans.fit_predict(vectors)

        cmap   = plt.cm.get_cmap("tab10")
        colors = [cmap(c / self.n_clusters) for c in cluster]

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, alpha=0.85, edgecolors="white", linewidths=0.5)

        # Підписуємо кожну точку
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (coords[i, 0], coords[i, 1]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

        # Легенда кластерів
        patches = [
            mpatches.Patch(color=cmap(c / self.n_clusters), label=f"Кластер {c+1}")
            for c in range(self.n_clusters)
        ]
        ax.legend(handles=patches, loc="best", fontsize=9)

        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% дисперсії)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% дисперсії)")
        ax.set_title(
            f"Кластеризація голів уваги (PCA + KMeans, {self.n_clusters} кластери)\n"
            "Кожна точка = одна голова (LxHy = шар x, голова y)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        save_or_show(fig, "09_head_clusters.png")

    def plot_cluster_avg_patterns(self, tokens, attentions):
        """
        Для кожного кластера — усереднена карта уваги всіх голів у ньому.
        """
        vectors, _ = self._extract_head_vectors(attentions)
        kmeans     = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster    = kmeans.fit_predict(vectors)

        seq_len = attentions[0].shape[-1]
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(5 * self.n_clusters, 5))

        import seaborn as sns
        for c in range(self.n_clusters):
            mask  = cluster == c
            heads = vectors[mask].reshape(-1, seq_len, seq_len).mean(axis=0)
            sns.heatmap(
                heads,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap=self.cfg.CMAP,
                cbar=False,
                ax=axes[c],
            )
            axes[c].set_title(f"Кластер {c+1}\n({mask.sum()} голів)", fontsize=10)
            axes[c].tick_params(axis="x", rotation=90)

        plt.suptitle(
            "Середній патерн уваги по кластерах",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "10_cluster_avg_patterns.png")
