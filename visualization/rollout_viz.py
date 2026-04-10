"""
Attention Rollout — накопичений вплив уваги через всі шари.

Звичайні карти показують лише один шар ізольовано.
Rollout перемножує матриці уваги через всі шари (з урахуванням
residual connections) і дає відповідь на питання:
"Який глобальний вплив має токен X на токен Y?"

Алгоритм (Abnar & Zuidema, 2020):
  R_0 = I
  R_i = R_{i-1} @ (0.5*A_i + 0.5*I)
де A_i — усереднена по головах матриця уваги шару i.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show


class RolloutVisualizer:
    def __init__(self, config):
        self.cfg = config

    def _compute_rollout(self, attentions) -> np.ndarray:
        """Повертає матрицю rollout (seq_len × seq_len)."""
        num_layers = len(attentions)
        seq_len = attentions[0].shape[-1]
        rollout = np.eye(seq_len)

        for layer_idx in range(num_layers):
            attn = attentions[layer_idx][0].mean(dim=0).numpy()
            # Додаємо residual connection: 0.5 * A + 0.5 * I
            attn_res = 0.5 * attn + 0.5 * np.eye(seq_len)
            # Нормалізуємо рядки
            attn_res = attn_res / attn_res.sum(axis=-1, keepdims=True)
            rollout = attn_res @ rollout

        return rollout

    def plot_rollout(self, tokens, attentions):
        """Будує теплову карту Attention Rollout."""
        rollout = self._compute_rollout(attentions)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            rollout,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap=self.cfg.CMAP,
            ax=ax,
        )
        ax.set_title(
            "Attention Rollout — накопичений вплив через всі шари\n"
            "(рядок = токен-джерело, стовпець = токен-ціль)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        save_or_show(fig, "07_attention_rollout.png")

    def plot_rollout_per_token(self, tokens, attentions):
        """
        Стовпчиковий графік: для кожного токена — його сумарний
        отриманий вплив (сума стовпця в rollout матриці).
        """
        rollout = self._compute_rollout(attentions)
        influence_scores = rollout.sum(axis=0)  # сума по рядках = хто на кого впливає

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.get_cmap(self.cfg.CMAP)(
            np.linspace(0.3, 0.9, len(tokens))
        )
        ax.bar(tokens, influence_scores, color=colors)
        ax.set_title(
            "Сумарний вплив кожного токена (Attention Rollout)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.set_ylabel("Сумарний вплив")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_or_show(fig, "08_rollout_per_token.png")
