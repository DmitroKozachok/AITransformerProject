"""
Representations — візуалізація hidden states токенів.

1. PCA/UMAP токенів — як змінюються вектори токенів від шару до шару
2. Cosine similarity між токенами на кожному шарі
3. Residual stream — внесок attention і FFN окремо
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _get_blocks_and_components(model):
    if hasattr(model, 'h'):
        return model.h, model.ln_f
    elif hasattr(model, 'transformer'):
        return model.transformer.h, model.transformer.ln_f
    raise AttributeError("Не вдалося знайти блоки GPT-2")


def _collect_hidden_states(model, tokenizer, text: str):
    """
    Повертає (tokens, hidden_states) де
    hidden_states — list з (num_layers+1) тензорів форми (seq, emb_dim).
    Індекс 0 = після embedding, індекс i = після i-го блоку.
    """
    inputs = tokenizer(text, return_tensors="pt")
    tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple з (num_layers+1) тензорів форми (1, seq, emb_dim)
    states = [h[0].numpy() for h in outputs.hidden_states]  # знімаємо batch-dim
    return tokens, states


def _collect_residual_contributions(model, tokenizer, text: str):
    """
    Повертає (tokens, attn_contribs, ffn_contribs) — внесок уваги і FFN
    на кожному шарі. Форма кожного: (num_layers, seq, emb_dim).
    """
    inputs = tokenizer(text, return_tensors="pt")
    tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    blocks, ln_f = _get_blocks_and_components(model)
    if hasattr(model, 'wte'):
        wte, wpe, drop = model.wte, model.wpe, model.drop
    else:
        wte, wpe, drop = model.transformer.wte, model.transformer.wpe, model.transformer.drop

    attn_contribs = []
    ffn_contribs  = []

    with torch.no_grad():
        pos_ids = torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)
        hidden  = wte(inputs["input_ids"]) + wpe(pos_ids)
        hidden  = drop(hidden)

        for block in blocks:
            # Внесок attention
            attn_out = block.attn(block.ln_1(hidden))[0]
            h_after_attn = hidden + attn_out
            attn_contribs.append(attn_out[0].norm(dim=-1).numpy())

            # Внесок FFN
            ffn_out = block.mlp(block.ln_2(h_after_attn))
            ffn_contribs.append(ffn_out[0].norm(dim=-1).numpy())

            hidden = h_after_attn + ffn_out

    return tokens, np.array(attn_contribs), np.array(ffn_contribs)


class RepresentationsVisualizer:
    def __init__(self, config):
        self.cfg = config

    # ------------------------------------------------------------------
    def plot_pca_layers(self, model, tokenizer, text: str):
        """PCA токенів: кожен шар у 2D, кольори = токени."""
        from sklearn.decomposition import PCA

        tokens, states = _collect_hidden_states(model, tokenizer, text)
        num_layers = len(states) - 1
        n_tokens   = len(tokens)

        cmap   = cm.get_cmap("tab10")
        colors = [cmap(i / n_tokens) for i in range(n_tokens)]

        cols = self.cfg.GRID_COLS
        rows = -(-( num_layers + 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=self.cfg.FIGURE_SIZE)
        axes = axes.flatten()

        pca = PCA(n_components=2)

        for i, state in enumerate(states):           # state: (seq, emb)
            pca.fit(state)
            coords = pca.transform(state)            # (seq, 2)
            ax = axes[i]
            for j, (tok, c) in enumerate(zip(tokens, colors)):
                ax.scatter(coords[j, 0], coords[j, 1], color=c, s=60, zorder=3)
                ax.annotate(tok, (coords[j, 0], coords[j, 1]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
            title = "Embedding" if i == 0 else f"Шар {i}"
            var = pca.explained_variance_ratio_
            ax.set_title(f"{title}\n({var[0]*100:.0f}%+{var[1]*100:.0f}%)", fontsize=9)
            ax.grid(True, alpha=0.2)

        for j in range(len(states), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("PCA представлень токенів по шарах", fontsize=self.cfg.FONT_SIZE_TITLE)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_or_show(fig, "16_pca_layers.png")

    # ------------------------------------------------------------------
    def plot_token_similarity(self, model, tokenizer, text: str):
        """Cosine similarity між токенами на кожному шарі."""
        tokens, states = _collect_hidden_states(model, tokenizer, text)
        num_layers = len(states) - 1

        cols = self.cfg.GRID_COLS
        rows = -(-(num_layers + 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=self.cfg.FIGURE_SIZE)
        axes = axes.flatten()

        for i, state in enumerate(states):
            norms  = np.linalg.norm(state, axis=1, keepdims=True)
            normed = state / (norms + 1e-9)
            sim    = normed @ normed.T

            sns.heatmap(sim, xticklabels=tokens, yticklabels=tokens,
                        cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                        ax=axes[i])
            title = "Embedding" if i == 0 else f"Шар {i}"
            axes[i].set_title(title, fontsize=9)
            axes[i].tick_params(axis="x", rotation=90, labelsize=7)
            axes[i].tick_params(axis="y", labelsize=7)

        for j in range(len(states), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Cosine similarity між токенами по шарах",
                     fontsize=self.cfg.FONT_SIZE_TITLE)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_or_show(fig, "17_token_similarity.png")

    # ------------------------------------------------------------------
    def plot_residual_stream(self, model, tokenizer, text: str):
        """Норма внеску attention і FFN на кожному шарі для кожного токена."""
        tokens, attn_c, ffn_c = _collect_residual_contributions(model, tokenizer, text)
        num_layers = len(attn_c)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        layer_labels = [f"L{i+1}" for i in range(num_layers)]

        # Attention contributions
        sns.heatmap(attn_c, xticklabels=tokens, yticklabels=layer_labels,
                    cmap=self.cfg.CMAP, ax=ax1)
        ax1.set_title("Внесок Attention\n(норма виходу)", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax1.tick_params(axis="x", rotation=90)

        # FFN contributions
        sns.heatmap(ffn_c, xticklabels=tokens, yticklabels=layer_labels,
                    cmap=self.cfg.CMAP, ax=ax2)
        ax2.set_title("Внесок FFN\n(норма виходу)", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax2.tick_params(axis="x", rotation=90)

        # Різниця: FFN - Attention
        diff = ffn_c - attn_c
        sns.heatmap(diff, xticklabels=tokens, yticklabels=layer_labels,
                    cmap="RdBu_r", center=0, ax=ax3)
        ax3.set_title("FFN − Attention\n(червоний = FFN домінує)", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax3.tick_params(axis="x", rotation=90)

        plt.suptitle("Residual Stream — внески компонентів", fontsize=self.cfg.FONT_SIZE_TITLE)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "18_residual_stream.png")

    def plot_similarity_per_layer(self, model, tokenizer, text: str):
        """Окреме вікно cosine similarity для кожного шару."""
        import os
        from utils.plot_utils import _output_dir

        tokens, states = _collect_hidden_states(model, tokenizer, text)
        out_dir = os.path.join(_output_dir, "similarity_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        for i, state in enumerate(states):
            norms  = np.linalg.norm(state, axis=1, keepdims=True)
            normed = state / (norms + 1e-9)
            sim    = normed @ normed.T

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(sim, xticklabels=tokens, yticklabels=tokens,
                        cmap="coolwarm", vmin=-1, vmax=1,
                        annot=True, fmt=".2f", ax=ax)
            title = "Embedding" if i == 0 else f"Шар {i}"
            ax.set_title(f"Cosine similarity — {title}",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            save_or_show(fig, f"sim_layer_{i:02d}.png", output_dir=out_dir)

    def plot_pca_per_layer(self, model, tokenizer, text: str):
        """Окремий вікно PCA для кожного шару."""
        from sklearn.decomposition import PCA
        import os
        from utils.plot_utils import _output_dir

        tokens, states = _collect_hidden_states(model, tokenizer, text)
        n_tokens = len(tokens)
        cmap   = cm.get_cmap("tab10")
        colors = [cmap(i / n_tokens) for i in range(n_tokens)]
        pca    = PCA(n_components=2)
        out_dir = os.path.join(_output_dir, "pca_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        for i, state in enumerate(states):
            pca.fit(state)
            coords = pca.transform(state)
            var    = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(7, 6))
            for j, (tok, c) in enumerate(zip(tokens, colors)):
                ax.scatter(coords[j, 0], coords[j, 1], color=c, s=80, zorder=3)
                ax.annotate(tok, (coords[j, 0], coords[j, 1]),
                            fontsize=9, xytext=(4, 4), textcoords="offset points")
            title = "Embedding" if i == 0 else f"Шар {i}"
            ax.set_title(f"PCA — {title}  ({var[0]*100:.0f}%+{var[1]*100:.0f}%)",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            save_or_show(fig, f"pca_layer_{i:02d}.png", output_dir=out_dir)

    def plot_residual_per_layer(self, model, tokenizer, text: str):
        """Окремий графік attention vs FFN для кожного шару."""
        import os
        from utils.plot_utils import _output_dir

        tokens, attn_c, ffn_c = _collect_residual_contributions(model, tokenizer, text)
        num_layers = len(attn_c)
        out_dir = os.path.join(_output_dir, "residual_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        x   = np.arange(len(tokens))
        w   = 0.35

        for layer_idx in range(num_layers):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(x - w/2, attn_c[layer_idx], w, label="Attention", color="steelblue", alpha=0.85)
            ax.bar(x + w/2, ffn_c[layer_idx],  w, label="FFN",       color="tomato",    alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha="right")
            ax.set_ylabel("Норма виходу")
            ax.set_title(f"Residual Stream — Шар {layer_idx+1}",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            save_or_show(fig, f"residual_layer_{layer_idx+1:02d}.png", output_dir=out_dir)