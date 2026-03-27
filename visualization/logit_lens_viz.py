"""
Logit Lens + Top-k передбачення по шарах.

"Підглядаємо" що модель передбачає після кожного шару —
не чекаючи фінального виходу. Показує як модель поступово
"вирішує" який токен наступний.
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _get_components(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    elif hasattr(model, 'transformer'):
        t = model.transformer
        return t.wte, t.wpe, t.drop, t.h, t.ln_f
    raise AttributeError("Не вдалося знайти компоненти GPT-2")


class LogitLensVisualizer:
    def __init__(self, config):
        self.cfg = config

    def _get_layer_logits(self, model, tokenizer, text: str):
        """
        Повертає (tokens, layer_logits) де
        layer_logits[i] — logits (seq, vocab) після i-го шару.

        Використовує output_hidden_states=True щоб уникнути ручного
        перебору блоків (block(hidden)[0] падає в нових версіях transformers).
        """
        inputs = tokenizer(text, return_tensors="pt")
        tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        wte, wpe, drop, blocks, ln_f = _get_components(model)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                output_hidden_states=True,
            )

        # hidden_states[0] = після embedding, [1..n] = після кожного блоку
        # Нас цікавлять тільки після блоків: [1:]
        layer_logits = []
        for hidden in outputs.hidden_states[1:]:
            # hidden: (1, seq, emb_dim)
            normed = ln_f(hidden)
            logits = normed @ wte.weight.T    # (1, seq, vocab)
            layer_logits.append(logits[0].detach().numpy())

        return tokens, layer_logits           # list of (seq, vocab)

    # ------------------------------------------------------------------
    def plot_logit_lens(self, model, tokenizer, text: str, top_k: int = 5):
        """
        Для кожної позиції і кожного шару: топ-1 передбачення наступного токена.
        Таблиця: рядки = шари, стовпці = позиції токенів.
        """
        tokens, layer_logits = self._get_layer_logits(model, tokenizer, text)
        num_layers = len(layer_logits)
        seq_len    = len(tokens)

        # Збираємо топ-1 токен і його ймовірність для кожної позиції/шару
        top1_tokens = []
        top1_probs  = np.zeros((num_layers, seq_len))

        for layer_idx, logits in enumerate(layer_logits):
            probs    = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            top_ids  = probs.argmax(axis=-1)   # (seq,)
            top_toks = clean_tokens(
                tokenizer.convert_ids_to_tokens(top_ids.tolist())
            )
            top1_tokens.append(top_toks)
            top1_probs[layer_idx] = probs[np.arange(seq_len), top_ids]

        # --- Heatmap ймовірностей ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        sns.heatmap(
            top1_probs,
            xticklabels=tokens,
            yticklabels=[f"L{i+1}" for i in range(num_layers)],
            cmap="YlOrRd", vmin=0, vmax=1,
            ax=ax1,
        )
        ax1.set_title("Logit Lens — впевненість топ-1 передбачення",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax1.tick_params(axis="x", rotation=45)

        # --- Текстова таблиця топ-1 токенів ---
        ax2.set_xlim(0, seq_len)
        ax2.set_ylim(0, num_layers)
        ax2.set_xticks(np.arange(seq_len) + 0.5)
        ax2.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax2.set_yticks(np.arange(num_layers) + 0.5)
        ax2.set_yticklabels([f"L{i+1}" for i in range(num_layers)], fontsize=9)
        ax2.invert_yaxis()

        cmap = plt.cm.get_cmap("YlOrRd")
        for layer_idx in range(num_layers):
            for pos in range(seq_len):
                prob  = top1_probs[layer_idx, pos]
                tok   = top1_tokens[layer_idx][pos]
                color = cmap(prob)
                ax2.add_patch(plt.Rectangle((pos, layer_idx), 1, 1, color=color))
                ax2.text(pos + 0.5, layer_idx + 0.5, tok,
                         ha="center", va="center", fontsize=7,
                         color="black" if prob < 0.7 else "white")

        ax2.set_title("Топ-1 передбачуваний токен на кожному шарі",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)

        plt.suptitle(f'Logit Lens: "{text}"', fontsize=self.cfg.FONT_SIZE_TITLE)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_or_show(fig, "19_logit_lens.png")

    # ------------------------------------------------------------------
    def plot_topk_per_layer(self, model, tokenizer, text: str,
                            position: int = -1, top_k: int = 10):
        """
        Для конкретної позиції токена: топ-k передбачень на кожному шарі.
        Показує як "думка" моделі про наступний токен змінюється по шарах.
        """
        tokens, layer_logits = self._get_layer_logits(model, tokenizer, text)
        seq_len    = len(tokens)
        num_layers = len(layer_logits)
        pos        = position % seq_len

        # Збираємо топ-k для кожного шару на позиції pos
        layer_topk = []
        for logits in layer_logits:
            probs   = torch.softmax(torch.tensor(logits[pos]), dim=-1).numpy()
            top_ids = probs.argsort()[-top_k:][::-1]
            top_toks = clean_tokens(tokenizer.convert_ids_to_tokens(top_ids.tolist()))
            layer_topk.append(list(zip(top_toks, probs[top_ids])))

        # Будуємо матрицю: рядки = шари, стовпці = топ-k токени
        all_toks = []
        for row in layer_topk:
            for tok, _ in row:
                if tok not in all_toks:
                    all_toks.append(tok)

        matrix = np.zeros((num_layers, len(all_toks)))
        for l_idx, row in enumerate(layer_topk):
            for tok, prob in row:
                t_idx = all_toks.index(tok)
                matrix[l_idx, t_idx] = prob

        fig, ax = plt.subplots(figsize=(max(10, len(all_toks) * 0.8), 7))
        sns.heatmap(
            matrix,
            xticklabels=all_toks,
            yticklabels=[f"L{i+1}" for i in range(num_layers)],
            cmap="YlOrRd", vmin=0,
            annot=True, fmt=".2f",
            ax=ax,
        )
        ax.set_title(
            f"Топ-{top_k} передбачень після токена '{tokens[pos]}' по шарах",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_or_show(fig, f"20_topk_predictions_pos{pos}.png")

    def plot_per_layer(self, model, tokenizer, text: str, top_k: int = 5):
        """
        Окреме вікно для кожного шару: топ-k передбачень для всіх позицій.
        """
        import os
        from utils.plot_utils import _output_dir

        tokens, layer_logits = self._get_layer_logits(model, tokenizer, text)
        seq_len    = len(tokens)
        out_dir    = os.path.join(_output_dir, "logit_lens_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        for layer_idx, logits in enumerate(layer_logits):
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

            fig, axes = plt.subplots(1, seq_len, figsize=(3 * seq_len, 5))
            if seq_len == 1:
                axes = [axes]

            for pos, ax in enumerate(axes):
                top_ids  = probs[pos].argsort()[-top_k:][::-1]
                top_prob = probs[pos][top_ids]
                top_toks = clean_tokens(
                    tokenizer.convert_ids_to_tokens(top_ids.tolist())
                )
                cmap   = plt.cm.get_cmap(self.cfg.CMAP)
                colors = [cmap(p) for p in top_prob]
                ax.barh(range(top_k), top_prob[::-1], color=colors[::-1])
                ax.set_yticks(range(top_k))
                ax.set_yticklabels(top_toks[::-1], fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_title(f"'{tokens[pos]}'", fontsize=9)
                ax.grid(axis="x", alpha=0.3)

            plt.suptitle(f"Logit Lens — Шар {layer_idx+1}  (топ-{top_k})",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            save_or_show(fig, f"logit_layer_{layer_idx+1:02d}.png", output_dir=out_dir)