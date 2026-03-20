"""
Візуалізація результатів детектора відповідей.

1. Feature heatmap       — всі 10 ознак для кожного токена відповіді
2. Token verdict bar     — колірна смужка міток CONFIDENT/UNCERTAIN/SUSPICIOUS
3. Radar chart           — середній профіль ознак відповіді
4. Per-feature timeline  — як кожна ознака змінюється по токенах
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from utils.plot_utils import save_or_show

COLORS = {
    "CONFIDENT":  "#2ecc71",
    "UNCERTAIN":  "#f39c12",
    "SUSPICIOUS": "#e74c3c",
}


class DetectorVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_feature_heatmap(self, analysis: dict):
        """Теплова карта: токени × 10 ознак."""
        features = analysis["features"]           # (seq, 10)
        tokens   = analysis["tokens"]
        names    = analysis["feature_names"]
        labels   = analysis["token_labels"]

        # Нормалізуємо кожну ознаку до [0,1] для порівняння
        normed = features.copy()
        for col in range(normed.shape[1]):
            mn, mx = normed[:, col].min(), normed[:, col].max()
            normed[:, col] = (normed[:, col] - mn) / (mx - mn + 1e-9)

        fig, (ax_map, ax_labels) = plt.subplots(
            2, 1, figsize=(max(10, len(tokens) * 0.9), 7),
            gridspec_kw={"height_ratios": [8, 1]}
        )

        sns.heatmap(
            normed.T,
            xticklabels=tokens,
            yticklabels=names,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            linewidths=0.3,
            ax=ax_map,
        )
        ax_map.set_title(
            f"Feature Map відповіді  |  Вердикт: {analysis['verdict']}",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax_map.tick_params(axis="x", rotation=45)

        # Кольорова смужка міток
        for i, lbl in enumerate(labels):
            ax_labels.add_patch(
                plt.Rectangle((i, 0), 1, 1, color=COLORS.get(lbl, "gray"))
            )
            ax_labels.text(i + 0.5, 0.5, lbl[0],
                           ha="center", va="center", fontsize=8, color="white",
                           fontweight="bold")
        ax_labels.set_xlim(0, len(labels))
        ax_labels.set_ylim(0, 1)
        ax_labels.axis("off")

        legend = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
        ax_labels.legend(handles=legend, loc="center right",
                         bbox_to_anchor=(1.12, 0.5), fontsize=8)

        plt.tight_layout()
        save_or_show(fig, "detector_01_feature_heatmap.png")

    def plot_verdict_bar(self, analysis: dict):
        """Детальний стовпчиковий графік по токенах з міткою і confidence."""
        tokens   = analysis["tokens"]
        labels   = analysis["token_labels"]
        features = analysis["features"]
        idx_conf = analysis["feature_names"].index("logit_confidence")
        conf     = features[:, idx_conf]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(tokens)), 6),
                                        gridspec_kw={"height_ratios": [3, 1]})

        colors = [COLORS.get(l, "gray") for l in labels]
        bars   = ax1.bar(tokens, conf, color=colors, alpha=0.85)
        ax1.set_ylim(0, 1.15)
        ax1.set_ylabel("Logit Confidence")
        ax1.axhline(0.7, color="green",  linestyle="--", alpha=0.5, label="Confident threshold")
        ax1.axhline(0.3, color="orange", linestyle="--", alpha=0.5, label="Uncertain threshold")
        ax1.legend(fontsize=8)
        ax1.set_title(f"Вердикт по токенах: {analysis['verdict']}\n"
                      f"({analysis['scores']})",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, conf):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=8)

        # Мітки
        for i, lbl in enumerate(labels):
            ax2.add_patch(plt.Rectangle((i, 0), 1, 1, color=COLORS.get(lbl, "gray")))
            ax2.text(i + 0.5, 0.5, lbl[:3], ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")
        ax2.set_xlim(0, len(labels))
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        plt.tight_layout()
        save_or_show(fig, "detector_02_verdict_bar.png")

    def plot_radar(self, analysis: dict):
        """Radar chart — середній профіль 10 ознак для відповіді."""
        features = analysis["features"]
        names    = analysis["feature_names"]
        n        = len(names)

        means = features.mean(axis=0)
        # Нормалізуємо до [0,1]
        mn, mx = means.min(), means.max()
        means_n = (means - mn) / (mx - mn + 1e-9)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values = means_n.tolist()
        # Замикаємо
        angles += angles[:1]
        values += values[:1]
        labels_closed = names + [names[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.plot(angles, values, "o-", linewidth=2, color="tomato")
        ax.fill(angles, values, alpha=0.25, color="tomato")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Feature Radar — відповідь\nВердикт: {analysis['verdict']}",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE, pad=20,
        )
        plt.tight_layout()
        save_or_show(fig, "detector_03_radar.png")

    def plot_feature_timeline(self, analysis: dict):
        """Лінійний графік кожної ознаки по токенах."""
        features = analysis["features"]     # (seq, 10)
        tokens   = analysis["tokens"]
        names    = analysis["feature_names"]
        labels   = analysis["token_labels"]
        n_feat   = len(names)

        cols = 2
        rows = -(-n_feat // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
        axes = axes.flatten()

        bg_colors = {"CONFIDENT": "#d5f5e3", "UNCERTAIN": "#fef9e7", "SUSPICIOUS": "#fadbd8"}
        x = np.arange(len(tokens))

        for i, (name, ax) in enumerate(zip(names, axes)):
            # Фон по мітках
            for j, lbl in enumerate(labels):
                ax.axvspan(j - 0.5, j + 0.5, color=bg_colors.get(lbl, "white"), alpha=0.4)

            ax.plot(x, features[:, i], marker="o", linewidth=2, color="steelblue")
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_title(name, fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        for j in range(n_feat, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Timeline по токенах відповіді",
                     fontsize=self.cfg.FONT_SIZE_TITLE)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_or_show(fig, "detector_04_feature_timeline.png")

    def plot_all(self, analysis: dict):
        """Будує всі 4 графіки детектора."""
        self.plot_feature_heatmap(analysis)
        self.plot_verdict_bar(analysis)
        self.plot_radar(analysis)
        self.plot_feature_timeline(analysis)