"""
Карта активацій — що "збуджується" в нейронах FFN шарів GPT-2.

GPT2Model з HuggingFace: блоки доступні через model.h (не model.transformer.h).
Чіпляємо forward hooks на mlp.act кожного блоку.
"""

import torch
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


class ActivationVisualizer:
    def __init__(self, config):
        self.cfg = config

    def _get_blocks(self, model):
        """
        Повертає список блоків GPT-2 незалежно від обгортки моделі.
        GPT2Model:        model.h
        GPT2LMHeadModel:  model.transformer.h
        """
        if hasattr(model, 'h'):
            return model.h
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        else:
            raise AttributeError(
                "Не вдалося знайти блоки GPT-2. "
                f"Доступні атрибути: {[n for n, _ in model.named_children()]}"
            )

    def _register_hooks(self, model):
        """Реєструємо forward hooks на FFN activation всіх блоків."""
        activations = {}
        hooks = []

        for layer_idx, block in enumerate(self._get_blocks(model)):
            def make_hook(idx):
                def hook(module, input, output):
                    activations[idx] = output.detach()
                return hook
            h = block.mlp.act.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        return activations, hooks

    def _remove_hooks(self, hooks):
        for h in hooks:
            h.remove()

    def get_activations(self, model, tokenizer, text: str):
        """Повертає (tokens, {layer_idx: np.ndarray(seq, ffn_dim)})."""
        inputs = tokenizer(text, return_tensors="pt")
        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

        activations, hooks = self._register_hooks(model)
        with torch.no_grad():
            model(**inputs)
        self._remove_hooks(hooks)

        result = {idx: act[0].numpy() for idx, act in activations.items()}
        return tokens, result

    def plot_all_layers(self, tokens, activations, max_neurons: int = 64):
        """Теплова карта для кожного шару: нейрони × токени."""
        num_layers = len(activations)
        cols = self.cfg.GRID_COLS
        rows = math.ceil(num_layers / cols)

        fig, axes = plt.subplots(rows, cols, figsize=self.cfg.FIGURE_SIZE)
        axes = axes.flatten()

        for layer_idx in range(num_layers):
            act = activations[layer_idx][:, :max_neurons]  # (seq, neurons)
            sns.heatmap(
                act.T,                   # нейрони по Y, токени по X
                xticklabels=tokens,
                yticklabels=False,
                cmap="RdBu_r",
                center=0,
                cbar=False,
                ax=axes[layer_idx],
            )
            axes[layer_idx].set_title(f"Шар {layer_idx+1}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            axes[layer_idx].tick_params(axis="x", rotation=90)

        for j in range(num_layers, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Активації FFN нейронів (перші {max_neurons}) — всі шари",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_or_show(fig, "12_ffn_activations_all_layers.png")

    def plot_top_neurons(self, tokens, activations, layer_idx: int, top_n: int = 20):
        """Топ-N найактивніших нейронів конкретного шару."""
        act = activations[layer_idx]
        neuron_max  = act.max(axis=0)
        top_indices = np.argsort(neuron_max)[-top_n:][::-1]
        top_act     = act[:, top_indices].T   # (top_n, seq)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            top_act,
            xticklabels=tokens,
            yticklabels=[f"N{i}" for i in top_indices],
            cmap="RdBu_r",
            center=0,
            ax=ax,
        )
        ax.set_title(
            f"Топ-{top_n} найактивніших нейронів FFN (Шар {layer_idx+1})",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        save_or_show(fig, f"13_top_neurons_layer{layer_idx+1}.png")

    def plot_per_layer(self, tokens, activations, max_neurons: int = 64):
        """Окреме вікно для кожного шару: топ нейрони × токени."""
        num_layers = len(activations)
        for layer_idx in range(num_layers):
            act = activations[layer_idx][:, :max_neurons].T  # (neurons, seq)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(act, xticklabels=tokens, yticklabels=False,
                        cmap="RdBu_r", center=0, cbar=True, ax=ax)
            ax.set_title(f"Активації FFN — Шар {layer_idx+1}",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.tick_params(axis="x", rotation=90)
            plt.tight_layout()
            save_or_show(fig, f"act_layer_{layer_idx+1:02d}.png",
                         output_dir=self._sublayer_dir("activations_per_layer"))

    @staticmethod
    def _sublayer_dir(name: str) -> str:
        import os
        from utils.plot_utils import _output_dir
        d = os.path.join(_output_dir, name)
        os.makedirs(d, exist_ok=True)
        return d
