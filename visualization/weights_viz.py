"""
Візуалізація ваг моделі.

1. Норми ваг по шарах
2. SVD матриць W_Q, W_K, W_V, W_O — які "концепти" закодовані
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show


def _get_blocks(model):
    if hasattr(model, 'h'):
        return model.h
    elif hasattr(model, 'transformer'):
        return model.transformer.h
    raise AttributeError("Не вдалося знайти блоки GPT-2")


class WeightsVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_weight_norms(self, model):
        """
        Норми ваг кожного компонента по шарах:
        W_Q, W_K, W_V, W_O (attention) + W_fc, W_proj (FFN).
        """
        blocks = _get_blocks(model)
        num_layers = len(blocks)

        components = {
            "W_Q":    [],
            "W_K":    [],
            "W_V":    [],
            "W_O":    [],
            "FFN_fc": [],
            "FFN_proj": [],
        }

        for block in blocks:
            # GPT-2 зберігає Q,K,V разом у c_attn (3*emb_dim)
            qkv_w = block.attn.c_attn.weight.detach()   # (emb, 3*emb)
            emb   = qkv_w.shape[0]
            wq, wk, wv = qkv_w[:, :emb], qkv_w[:, emb:2*emb], qkv_w[:, 2*emb:]
            wo    = block.attn.c_proj.weight.detach()

            components["W_Q"].append(wq.norm().item())
            components["W_K"].append(wk.norm().item())
            components["W_V"].append(wv.norm().item())
            components["W_O"].append(wo.norm().item())
            components["FFN_fc"].append(block.mlp.c_fc.weight.detach().norm().item())
            components["FFN_proj"].append(block.mlp.c_proj.weight.detach().norm().item())

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        layer_labels = [f"L{i+1}" for i in range(num_layers)]
        colors = plt.cm.get_cmap(self.cfg.CMAP)(np.linspace(0.3, 0.9, num_layers))

        for ax, (name, norms) in zip(axes, components.items()):
            ax.bar(layer_labels, norms, color=colors)
            ax.set_title(f"Норма {name}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.set_xlabel("Шар")
            ax.set_ylabel("‖W‖")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Норми ваг по шарах", fontsize=self.cfg.FONT_SIZE_TITLE)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_or_show(fig, "21_weight_norms.png")

    def plot_svd_weights(self, model, layer_idx: int, top_k: int = 20):
        """
        SVD матриць W_Q, W_K, W_V, W_O конкретного шару.
        Сингулярні значення показують "ємність" кожного напрямку.
        """
        blocks = _get_blocks(model)
        block  = blocks[layer_idx]

        qkv_w = block.attn.c_attn.weight.detach().numpy()
        emb   = qkv_w.shape[0]
        matrices = {
            "W_Q": qkv_w[:, :emb],
            "W_K": qkv_w[:, emb:2*emb],
            "W_V": qkv_w[:, 2*emb:],
            "W_O": block.attn.c_proj.weight.detach().numpy(),
        }

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        cmap = plt.cm.get_cmap(self.cfg.CMAP)

        for ax, (name, W) in zip(axes, matrices.items()):
            _, sv, _ = np.linalg.svd(W, full_matrices=False)
            sv_top = sv[:top_k]
            colors = [cmap(i / top_k) for i in range(top_k)]

            ax.bar(range(top_k), sv_top, color=colors)
            ax.set_title(f"SVD {name}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.set_xlabel("Компонента")
            ax.set_ylabel("Сингулярне значення")
            ax.grid(axis="y", alpha=0.3)

            # Підписуємо скільки % дисперсії покривають топ-k
            total_var = (sv ** 2).sum()
            top_var   = (sv_top ** 2).sum()
            ax.text(0.98, 0.97, f"топ-{top_k}: {top_var/total_var*100:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        plt.suptitle(
            f"SVD матриць уваги (Шар {layer_idx+1})",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, f"22_svd_weights_layer{layer_idx+1}.png")

    def plot_weight_heatmap(self, model, layer_idx: int):
        """
        Теплові карти самих матриць W_Q, W_K, W_V, W_O (зменшені через PCA).
        """
        from sklearn.decomposition import PCA

        blocks = _get_blocks(model)
        block  = blocks[layer_idx]

        qkv_w = block.attn.c_attn.weight.detach().numpy()
        emb   = qkv_w.shape[0]
        matrices = {
            "W_Q": qkv_w[:, :emb],
            "W_K": qkv_w[:, emb:2*emb],
            "W_V": qkv_w[:, 2*emb:],
            "W_O": block.attn.c_proj.weight.detach().numpy(),
        }

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (name, W) in zip(axes, matrices.items()):
            # Зменшуємо до 32×32 через PCA для відображення
            n = min(32, W.shape[0], W.shape[1])
            pca_r = PCA(n_components=n).fit_transform(W)[:n, :n]
            sns.heatmap(pca_r, cmap="RdBu_r", center=0, cbar=True,
                        xticklabels=False, yticklabels=False, ax=ax)
            ax.set_title(f"{name} (PCA 32×32)", fontsize=self.cfg.FONT_SIZE_SUBTITLE)

        plt.suptitle(
            f"Структура матриць уваги (Шар {layer_idx+1})",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, f"23_weight_heatmap_layer{layer_idx+1}.png")


    def plot_svd_per_layer(self, model, top_k: int = 20):
        """SVD матриць W_Q, W_K, W_V, W_O для кожного шару окремо."""
        import os
        from utils.plot_utils import _output_dir

        blocks  = _get_blocks(model)
        out_dir = os.path.join(_output_dir, "svd_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        for layer_idx, block in enumerate(blocks):
            qkv_w = block.attn.c_attn.weight.detach().numpy()
            emb   = qkv_w.shape[0]
            matrices = {
                "W_Q": qkv_w[:, :emb],
                "W_K": qkv_w[:, emb:2*emb],
                "W_V": qkv_w[:, 2*emb:],
                "W_O": block.attn.c_proj.weight.detach().numpy(),
            }

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            cmap = plt.cm.get_cmap(self.cfg.CMAP)

            for ax, (name, W) in zip(axes, matrices.items()):
                _, sv, _ = np.linalg.svd(W, full_matrices=False)
                sv_top   = sv[:top_k]
                colors   = [cmap(i / top_k) for i in range(top_k)]
                ax.bar(range(top_k), sv_top, color=colors)
                ax.set_title(f"SVD {name}", fontsize=10)
                ax.set_xlabel("Компонента")
                ax.set_ylabel("Сингулярне значення")
                ax.grid(axis="y", alpha=0.3)
                total_var = (sv ** 2).sum()
                top_var   = (sv_top ** 2).sum()
                ax.text(0.98, 0.97, f"топ-{top_k}: {top_var/total_var*100:.1f}%",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

            plt.suptitle(f"SVD матриць уваги — Шар {layer_idx+1}",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            save_or_show(fig, f"svd_layer_{layer_idx+1:02d}.png", output_dir=out_dir)

    def plot_weight_heatmap_per_layer(self, model):
        """Теплові карти матриць ваг для кожного шару."""
        from sklearn.decomposition import PCA
        import os
        from utils.plot_utils import _output_dir

        blocks  = _get_blocks(model)
        out_dir = os.path.join(_output_dir, "weight_heatmaps_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        for layer_idx, block in enumerate(blocks):
            qkv_w = block.attn.c_attn.weight.detach().numpy()
            emb   = qkv_w.shape[0]
            matrices = {
                "W_Q": qkv_w[:, :emb],
                "W_K": qkv_w[:, emb:2*emb],
                "W_V": qkv_w[:, 2*emb:],
                "W_O": block.attn.c_proj.weight.detach().numpy(),
            }

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for ax, (name, W) in zip(axes, matrices.items()):
                n = min(32, W.shape[0], W.shape[1])
                pca_r = PCA(n_components=n).fit_transform(W)[:n, :n]
                sns.heatmap(pca_r, cmap="RdBu_r", center=0, cbar=True,
                            xticklabels=False, yticklabels=False, ax=ax)
                ax.set_title(f"{name}", fontsize=10)

            plt.suptitle(f"Матриці ваг — Шар {layer_idx+1}",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            save_or_show(fig, f"weights_layer_{layer_idx+1:02d}.png", output_dir=out_dir)