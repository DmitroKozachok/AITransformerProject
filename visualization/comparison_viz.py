"""
Порівняння карт уваги для різних речень
"""

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show


def _clean_tokens(tokens):
    """
    GPT-2 позначає пробіл перед словом символом Ġ (Unicode U+0120).
    Замінюємо його на '·' щоб підписи були читабельними.
    """
    return [t.replace("\u0120", "·").replace("Ġ", "·") for t in tokens]


class ComparisonVisualizer:
    def __init__(self, model, tokenizer, config):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = config

    def _get_attention_for_text(self, text: str, layer_idx: int):
        """Повертає (clean_tokens, avg_attention_matrix) для заданого тексту та шару."""
        inputs = self.tokenizer(text, return_tensors="pt")
        raw_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens = _clean_tokens(raw_tokens)

        with torch.no_grad():
            outputs = self.model(**inputs)

        avg_attn = outputs.attentions[layer_idx][0].mean(dim=0).numpy()
        return tokens, avg_attn

    def plot_comparison(self, texts: list[str], layer_idx: int):
        """Малює карти уваги для кожного тексту поруч."""
        n = len(texts)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

        if n == 1:
            axes = [axes]

        for i, text in enumerate(texts):
            tokens, attn = self._get_attention_for_text(text, layer_idx)

            sns.heatmap(
                attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap=self.cfg.CMAP,
                cbar=False,
                ax=axes[i],
            )
            short = text if len(text) <= 35 else text[:32] + "..."
            axes[i].set_title(f'"{short}"', fontsize=9)
            axes[i].tick_params(axis="x", rotation=90)

        plt.suptitle(
            f"Порівняння карт уваги (Шар {layer_idx + 1})",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, f"05_comparison_layer{layer_idx + 1}.png")