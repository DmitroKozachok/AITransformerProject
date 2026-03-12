"""
Теплові карти уваги (усереднені та по кожній голові)
"""

import math
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show


class HeatmapVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_all_layers(self, tokens, attentions):
        """Усереднені карти уваги для кожного шару."""
        num_layers = len(attentions)
        cols = self.cfg.GRID_COLS
        rows = math.ceil(num_layers / cols)

        fig, axes = plt.subplots(rows, cols, figsize=self.cfg.FIGURE_SIZE)
        axes = axes.flatten()

        for i in range(num_layers):
            layer_attn = attentions[i][0].mean(dim=0).numpy()
            sns.heatmap(
                layer_attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap=self.cfg.CMAP,
                cbar=False,
                ax=axes[i],
            )
            axes[i].set_title(f"Шар {i + 1}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            axes[i].tick_params(axis="x", rotation=90)

        for j in range(num_layers, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "Карти уваги (усереднені) — всі шари GPT-2",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_or_show(fig, "01_all_layers_heatmap.png")

    def plot_all_heads(self, tokens, attentions, layer_idx: int):
        """Карти уваги для кожної голови в одному шарі."""
        num_heads = attentions[layer_idx].shape[1]
        cols = self.cfg.GRID_COLS
        rows = math.ceil(num_heads / cols)

        fig, axes = plt.subplots(rows, cols, figsize=self.cfg.FIGURE_SIZE)
        axes = axes.flatten()

        for h in range(num_heads):
            head_attn = attentions[layer_idx][0][h].numpy()
            sns.heatmap(
                head_attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap=self.cfg.CMAP,
                cbar=False,
                ax=axes[h],
            )
            axes[h].set_title(f"Голова {h + 1}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            axes[h].tick_params(axis="x", rotation=90)

        for j in range(num_heads, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Шар {layer_idx + 1} — всі голови уваги",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_or_show(fig, f"02_layer{layer_idx + 1}_all_heads.png")