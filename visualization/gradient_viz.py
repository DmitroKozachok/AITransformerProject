"""
Карта градієнтів — Gradient × Input saliency.

Три методи важливості токенів:
  - Gradient Saliency:    |∂loss/∂embed|
  - Gradient × Input:     |grad ⊙ embed|
  - Integrated Gradients: середнє градієнтів вздовж шляху від нуля
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _get_embed_layer(model):
    """Повертає wte і wpe незалежно від обгортки моделі."""
    if hasattr(model, 'wte'):                          # GPT2Model
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    elif hasattr(model, 'transformer'):                # GPT2LMHeadModel
        t = model.transformer
        return t.wte, t.wpe, t.drop, t.h, t.ln_f
    else:
        raise AttributeError("Не вдалося визначити структуру моделі GPT-2")


def _lm_head(model):
    """Повертає lm_head якщо є, інакше None."""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    return None


def _forward_from_embeds(embeds_with_grad, model, seq_len):
    """
    Проводить forward pass починаючи з готових embeddings.
    Повертає logits (1, seq, vocab) або hidden (1, seq, emb_dim).
    """
    wte, wpe, drop, blocks, ln_f = _get_embed_layer(model)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    hidden = embeds_with_grad + wpe(position_ids)
    hidden = drop(hidden)
    for block in blocks:
        hidden = block(hidden)[0]
    hidden = ln_f(hidden)

    lm = _lm_head(model)
    if lm is not None:
        return lm(hidden)
    # GPT2Model не має lm_head — використовуємо wte як проекцію
    return hidden @ wte.weight.T


class GradientVisualizer:
    def __init__(self, config):
        self.cfg = config

    def _compute_gradients(self, model, tokenizer, text: str):
        """Повертає (tokens, grad_saliency, grad_x_input)."""
        inputs  = tokenizer(text, return_tensors="pt")
        tokens  = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        seq_len = inputs["input_ids"].shape[1]

        wte = _get_embed_layer(model)[0]
        real_embeds = wte(inputs["input_ids"]).detach()   # (1, seq, emb_dim)
        embeds_with_grad = real_embeds.clone().requires_grad_(True)

        if seq_len < 2:
            return tokens, np.zeros(seq_len), np.zeros(seq_len)

        model.eval()
        logits = _forward_from_embeds(embeds_with_grad, model, seq_len)

        shift_logits = logits[0, :-1, :]
        shift_labels = inputs["input_ids"][0, 1:]
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        loss.backward()

        grad = embeds_with_grad.grad[0]   # (seq, emb_dim)
        grad_saliency = grad.norm(dim=-1).detach().numpy()
        grad_x_input  = (grad * embeds_with_grad[0]).norm(dim=-1).detach().numpy()

        return tokens, grad_saliency, grad_x_input

    def _compute_integrated_gradients(self, model, tokenizer, text: str, steps: int = 20):
        """Integrated Gradients: усереднення градієнтів вздовж лінійного шляху."""
        inputs  = tokenizer(text, return_tensors="pt")
        tokens  = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        seq_len = inputs["input_ids"].shape[1]

        wte = _get_embed_layer(model)[0]
        baseline    = torch.zeros(1, seq_len, wte.embedding_dim)
        real_embeds = wte(inputs["input_ids"]).detach()

        if seq_len < 2:
            return tokens, np.zeros(seq_len)

        integrated_grads = torch.zeros_like(real_embeds[0])
        model.eval()

        for step in range(steps):
            alpha  = step / steps
            interp = (baseline + alpha * (real_embeds - baseline)).requires_grad_(True)

            logits = _forward_from_embeds(interp, model, seq_len)
            shift_logits = logits[0, :-1, :]
            shift_labels = inputs["input_ids"][0, 1:]
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
            loss.backward()

            integrated_grads += interp.grad[0].detach()

        integrated_grads /= steps
        ig_scores = (integrated_grads * (real_embeds[0] - baseline[0])).norm(dim=-1).numpy()
        return tokens, ig_scores

    def plot_saliency_comparison(self, model, tokenizer, text: str):
        """Три методи на одному графіку."""
        tokens, grad_sal, grad_x_inp = self._compute_gradients(model, tokenizer, text)
        _,      ig_scores            = self._compute_integrated_gradients(model, tokenizer, text)

        def norm(x):
            r = x - x.min()
            return r / (r.max() + 1e-9)

        methods = {
            "Gradient Saliency  |∂loss/∂embed|":  norm(grad_sal),
            "Gradient × Input   |grad ⊙ embed|":  norm(grad_x_inp),
            "Integrated Gradients  (20 кроків)":  norm(ig_scores),
        }

        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        cmap = plt.cm.get_cmap(self.cfg.CMAP)

        for ax, (title, scores) in zip(axes, methods.items()):
            colors = [cmap(s) for s in scores]
            bars = ax.bar(tokens, scores, color=colors)
            ax.set_title(title, fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Важливість (норм.)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        plt.suptitle(
            f"Градієнтна важливість токенів\n\"{text}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        save_or_show(fig, "14_gradient_saliency.png")

    def plot_gradient_heatmap(self, model, tokenizer, texts: list[str]):
        """Heatmap важливості токенів між реченнями (Gradient×Input)."""
        all_scores, all_tokens, max_len = [], [], 0

        for text in texts:
            tokens, _, gxi = self._compute_gradients(model, tokenizer, text)
            all_scores.append(gxi)
            all_tokens.append(tokens)
            max_len = max(max_len, len(tokens))

        col_labels = [f"pos{i}" for i in range(max_len)]
        matrix     = np.zeros((len(texts), max_len))

        for i, (scores, tokens) in enumerate(zip(all_scores, all_tokens)):
            matrix[i, :len(scores)] = scores
            if i == 0:
                col_labels[:len(tokens)] = tokens

        row_max = matrix.max(axis=1, keepdims=True)
        matrix  = matrix / (row_max + 1e-9)

        row_labels = [t[:30] + "..." if len(t) > 30 else t for t in texts]

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(
            matrix,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap=self.cfg.CMAP,
            vmin=0, vmax=1,
            ax=ax,
        )
        ax.set_title(
            "Порівняння Gradient×Input важливості між реченнями",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_or_show(fig, "15_gradient_heatmap_comparison.png")

    def plot_per_layer(self, model, tokenizer, text: str):
        """
        Для кожного шару — градієнт по embedding після i шарів.
        Використовуємо forward hooks щоб отримати gradient saliency
        на виході кожного конкретного шару.
        """
        from utils.token_utils import clean_tokens

        inputs  = tokenizer(text, return_tensors="pt")
        tokens  = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        seq_len = inputs["input_ids"].shape[1]

        if hasattr(model, 'h'):
            wte, wpe, drop, blocks, ln_f = model.wte, model.wpe, model.drop, model.h, model.ln_f
        else:
            t = model.transformer
            wte, wpe, drop, blocks, ln_f = t.wte, t.wpe, t.drop, t.h, t.ln_f

        if seq_len < 2:
            return

        import os
        from utils.plot_utils import _output_dir

        out_dir = os.path.join(_output_dir, "gradients_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        for stop_layer in range(len(blocks)):
            embeds = wte(inputs["input_ids"]).detach().clone().requires_grad_(True)
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            hidden = embeds + wpe(pos_ids)
            hidden = drop(hidden)

            for idx, block in enumerate(blocks):
                hidden = block(hidden)[0]
                if idx == stop_layer:
                    break

            hidden = ln_f(hidden)
            logits = hidden @ wte.weight.T
            loss   = torch.nn.functional.cross_entropy(
                logits[0, :-1], inputs["input_ids"][0, 1:]
            )
            loss.backward()

            scores = embeds.grad[0].norm(dim=-1).detach().numpy()
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

            fig, ax = plt.subplots(figsize=(9, 3))
            cmap   = plt.cm.get_cmap(self.cfg.CMAP)
            colors = [cmap(s) for s in scores]
            bars   = ax.bar(tokens, scores, color=colors)
            for bar, val in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", fontsize=8)
            ax.set_ylim(0, 1.2)
            ax.set_title(f"Gradient Saliency — Шар {stop_layer+1}",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            save_or_show(fig, f"grad_layer_{stop_layer+1:02d}.png", output_dir=out_dir)
