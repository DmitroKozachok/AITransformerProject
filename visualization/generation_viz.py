"""
Generation Viz — покроковий процес генерації.

Показує кожен крок генерації окремо:
- який токен обрано
- топ-K альтернативи
- впевненість по шарах
- на що дивилась увага в момент вибору

1. generation_stream   — всі токени як потік з ймовірностями
2. step_by_step        — кожен крок генерації окремою панеллю
3. generation_heatmap  — зведена теплова карта всіх кроків
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _get_parts(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    t = model.transformer
    return t.wte, t.wpe, t.drop, t.h, t.ln_f


def _generate_with_details(model, tokenizer, question: str,
                            max_new_tokens: int = 20,
                            top_k_show: int = 5) -> dict:
    """
    Генерує відповідь токен за токеном і для кожного кроку зберігає:
    - обраний токен і його ймовірність
    - топ-K альтернативи
    - увагу на всі попередні токени
    - ймовірності по шарах (logit lens)
    """
    wte, wpe, drop, blocks, ln_f = _get_parts(model)
    num_layers = len(blocks)
    eos_id     = tokenizer.eos_token_id

    inputs   = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    q_len     = len(input_ids)

    steps = []  # список кроків генерації

    generated = input_ids[:]

    for step in range(max_new_tokens):
        ids_t   = torch.tensor([generated])
        seq_len = ids_t.shape[1]

        # Збираємо logit lens і увагу
        layer_top1_tok  = []
        layer_top1_prob = []
        final_attn      = None

        with torch.no_grad():
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            hidden  = wte(ids_t) + wpe(pos_ids)
            hidden  = drop(hidden)

            for layer_idx, block in enumerate(blocks):
                attn_out, attn_w = block.attn(block.ln_1(hidden))[:2]
                hidden = hidden + attn_out
                hidden = hidden + block.mlp(block.ln_2(hidden))

                # Logit lens для останньої позиції
                normed = ln_f(hidden)
                if hasattr(model, 'lm_head'):
                    logits = model.lm_head(normed)[0, -1]
                else:
                    logits = (normed[0, -1]) @ wte.weight.T

                probs   = torch.softmax(logits, dim=-1)
                top1_id = int(probs.argmax())
                top1_p  = float(probs[top1_id])
                top1_tok = clean_tokens(
                    tokenizer.convert_ids_to_tokens([top1_id])
                )[0]
                layer_top1_tok.append(top1_tok)
                layer_top1_prob.append(top1_p)

                if layer_idx == num_layers - 1:
                    final_attn = attn_w[0].mean(dim=0).numpy()  # (seq, seq)

        # Фінальне рішення (останній шар)
        with torch.no_grad():
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            h = wte(ids_t) + wpe(pos_ids)
            h = drop(h)
            for block in blocks:
                h = block(h)[0]
            h = ln_f(h)
            if hasattr(model, 'lm_head'):
                final_logits = model.lm_head(h)[0, -1]
            else:
                final_logits = h[0, -1] @ wte.weight.T

        probs_final = torch.softmax(final_logits, dim=-1)
        topk_ids    = probs_final.topk(top_k_show).indices.tolist()
        topk_probs  = probs_final.topk(top_k_show).values.tolist()
        topk_toks   = clean_tokens(tokenizer.convert_ids_to_tokens(topk_ids))

        chosen_id   = topk_ids[0]
        chosen_tok  = topk_toks[0]
        chosen_prob = topk_probs[0]

        # Контекст — всі токени до цього кроку
        context_toks = clean_tokens(
            tokenizer.convert_ids_to_tokens(generated)
        )

        steps.append({
            "step":          step + 1,
            "context":       context_toks,
            "chosen_tok":    chosen_tok,
            "chosen_prob":   chosen_prob,
            "topk_toks":     topk_toks,
            "topk_probs":    topk_probs,
            "layer_thoughts": layer_top1_tok,    # (num_layers,)
            "layer_probs":    layer_top1_prob,    # (num_layers,)
            "attention":      final_attn,          # (seq, seq)
        })

        generated.append(chosen_id)

        if chosen_id == eos_id:
            break

    # Фінальний текст
    answer_ids  = generated[q_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    q_tokens    = clean_tokens(tokenizer.convert_ids_to_tokens(input_ids))

    return {
        "question":    question,
        "answer":      answer_text.strip(),
        "q_tokens":    q_tokens,
        "q_len":       q_len,
        "steps":       steps,
        "num_layers":  num_layers,
        "top_k":       top_k_show,
    }


class GenerationVisualizer:
    def __init__(self, config):
        self.cfg = config

    # ── 1. Generation Stream ─────────────────────────────────────────
    def plot_generation_stream(self, data: dict):
        """
        Всі згенеровані токени як потік зліва направо.
        Колір = впевненість, висота = ймовірність.
        Альтернативи показані знизу кожного токена.
        """
        steps   = data["steps"]
        n_steps = len(steps)
        top_k   = data["top_k"]

        fig, (ax_main, ax_alt) = plt.subplots(
            2, 1, figsize=(max(14, n_steps * 1.4), 7),
            gridspec_kw={"height_ratios": [3, 2]}
        )

        cmap = plt.cm.get_cmap("RdYlGn")

        # Верхня панель: обрані токени
        for i, step in enumerate(steps):
            prob  = step["chosen_prob"]
            color = cmap(prob)
            bar   = ax_main.bar(i, prob, color=color, alpha=0.85,
                                edgecolor="white", linewidth=0.5)
            ax_main.text(i, prob + 0.01, step["chosen_tok"],
                         ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax_main.text(i, -0.06, f"{prob:.2f}",
                         ha="center", va="top", fontsize=7, color="gray")

        ax_main.set_xlim(-0.5, n_steps - 0.5)
        ax_main.set_ylim(-0.1, 1.2)
        ax_main.set_xticks(range(n_steps))
        ax_main.set_xticklabels([f"#{s['step']}" for s in steps], fontsize=8)
        ax_main.set_ylabel("Впевненість")
        ax_main.set_title("Потік Генерації — обрані токени і впевненість",
                          fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax_main.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax_main.grid(axis="y", alpha=0.2)

        # Нижня панель: альтернативи
        alt_colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
        for rank in range(min(top_k, 5)):
            ys = []
            for step in steps:
                if rank < len(step["topk_probs"]):
                    ys.append(step["topk_probs"][rank])
                else:
                    ys.append(0)
            label = f"Топ-{rank+1}"
            ax_alt.plot(range(n_steps), ys,
                        color=alt_colors[rank % len(alt_colors)],
                        linewidth=2 if rank == 0 else 1,
                        linestyle="-" if rank == 0 else "--",
                        alpha=0.9 if rank == 0 else 0.6,
                        label=label, marker="o", markersize=4)

            # Підписуємо токени для топ-2
            if rank < 2:
                for i, step in enumerate(steps):
                    if rank < len(step["topk_toks"]):
                        ax_alt.text(i, ys[i] + 0.01,
                                    step["topk_toks"][rank][:5],
                                    ha="center", fontsize=6,
                                    color=alt_colors[rank])

        ax_alt.set_xlim(-0.5, n_steps - 0.5)
        ax_alt.set_ylim(0, 1.1)
        ax_alt.set_xticks(range(n_steps))
        ax_alt.set_xticklabels([s["chosen_tok"] for s in steps],
                               rotation=45, ha="right", fontsize=8)
        ax_alt.set_ylabel("Ймовірність")
        ax_alt.set_title("Альтернативні токени на кожному кроці",
                         fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax_alt.legend(fontsize=8, loc="upper right")
        ax_alt.grid(alpha=0.2)

        plt.suptitle(
            f"Generation Stream\n"
            f"Q: \"{data['question']}\"\n"
            f"A: \"{data['answer'][:60]}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "gen_01_stream.png")

    # ── 2. Step by Step ──────────────────────────────────────────────
    def plot_step_by_step(self, data: dict):
        """
        Кожен крок генерації — окрема панель:
        ліво: топ-K кандидати | право: увага на контекст
        """
        steps      = data["steps"]
        num_layers = data["num_layers"]
        top_k      = data["top_k"]
        q_len      = data["q_len"]

        max_show = min(len(steps), 8)  # максимум 8 кроків
        fig = plt.figure(figsize=(16, max_show * 2.5))
        gs  = gridspec.GridSpec(max_show, 3,
                                hspace=0.5, wspace=0.35,
                                width_ratios=[2, 3, 2])

        cmap_attn = plt.cm.get_cmap("Blues")
        cmap_lay  = plt.cm.get_cmap("YlOrRd")

        for row, step in enumerate(steps[:max_show]):
            # ── Ліво: топ-K кандидати ──
            ax_bar = fig.add_subplot(gs[row, 0])
            toks   = step["topk_toks"]
            probs  = step["topk_probs"]
            colors = ["#2ecc71" if i == 0 else "#3498db"
                      for i in range(len(toks))]
            ax_bar.barh(range(len(toks)), probs[::-1],
                        color=colors[::-1], alpha=0.85)
            ax_bar.set_yticks(range(len(toks)))
            ax_bar.set_yticklabels(toks[::-1], fontsize=8)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_title(f"Крок {step['step']}: вибрано '{step['chosen_tok']}' "
                             f"({step['chosen_prob']:.2f})",
                             fontsize=8)
            ax_bar.grid(axis="x", alpha=0.3)

            # ── Центр: logit lens по шарах ──
            ax_lay = fig.add_subplot(gs[row, 1])
            layer_probs = step["layer_probs"]
            layer_toks  = step["layer_thoughts"]
            ax_lay.barh(range(num_layers), layer_probs,
                        color=[cmap_lay(p) for p in layer_probs])
            for l_idx, (lt, lp) in enumerate(zip(layer_toks, layer_probs)):
                ax_lay.text(lp + 0.01, l_idx, lt[:8],
                            va="center", fontsize=6)
            ax_lay.set_yticks(range(0, num_layers, 3))
            ax_lay.set_yticklabels([f"L{i+1}" for i in range(0, num_layers, 3)],
                                   fontsize=7)
            ax_lay.set_xlim(0, 1.3)
            ax_lay.set_title("Думка по шарах", fontsize=8)
            ax_lay.grid(axis="x", alpha=0.2)

            # ── Право: увага на контекст ──
            ax_attn = fig.add_subplot(gs[row, 2])
            attn    = step["attention"]  # (seq, seq)
            # Увага останнього токена на всі попередні
            last_attn   = attn[-1, :]
            context_tok = step["context"]
            show_n      = min(len(context_tok), 10)
            ax_attn.barh(range(show_n),
                         last_attn[-show_n:][::-1],
                         color=[cmap_attn(v) for v in last_attn[-show_n:][::-1]])
            ax_attn.set_yticks(range(show_n))
            ax_attn.set_yticklabels(context_tok[-show_n:][::-1], fontsize=7)
            ax_attn.set_title("Увага на контекст", fontsize=8)
            ax_attn.set_xlim(0, max(last_attn[-show_n:].max() * 1.3, 0.1))
            ax_attn.grid(axis="x", alpha=0.2)

        plt.suptitle(
            f"Step-by-Step Генерація\nQ: \"{data['question']}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        save_or_show(fig, "gen_02_step_by_step.png")

    # ── 3. Generation Heatmap ─────────────────────────────────────────
    def plot_generation_heatmap(self, data: dict):
        """
        Зведена теплова карта: кроки × шари.
        Колір = впевненість, текст = що думав шар.
        """
        steps      = data["steps"]
        num_layers = data["num_layers"]
        n_steps    = len(steps)

        matrix       = np.zeros((num_layers, n_steps))
        thought_matrix = [[""] * n_steps for _ in range(num_layers)]

        for step_idx, step in enumerate(steps):
            for l_idx, (p, t) in enumerate(zip(step["layer_probs"],
                                                step["layer_thoughts"])):
                matrix[l_idx, step_idx]          = p
                thought_matrix[l_idx][step_idx]  = t[:5]

        fig, ax = plt.subplots(figsize=(max(14, n_steps * 1.2), 8))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1)

        # Текст у клітинках
        for l_idx in range(num_layers):
            for s_idx in range(n_steps):
                p = matrix[l_idx, s_idx]
                t = thought_matrix[l_idx][s_idx]
                color = "white" if p > 0.6 else "black"
                ax.text(s_idx, l_idx, t,
                        ha="center", va="center", fontsize=6, color=color)

        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{i+1}" for i in range(num_layers)], fontsize=8)
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels(
            [f"#{s['step']}\n'{s['chosen_tok']}'\n{s['chosen_prob']:.2f}"
             for s in steps],
            fontsize=7,
        )

        plt.colorbar(im, ax=ax, fraction=0.02, label="Впевненість")
        ax.set_title(
            f"Generation Heatmap — думка кожного шару на кожному кроці\n"
            f"Q: \"{data['question']}\"  →  A: \"{data['answer'][:50]}\"",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        save_or_show(fig, "gen_03_heatmap.png")

    def plot_all(self, model, tokenizer, question: str,
                 max_new_tokens: int = 20, top_k: int = 5):
        print("  Генеруємо токен за токеном...")
        data = _generate_with_details(
            model, tokenizer, question,
            max_new_tokens=max_new_tokens,
            top_k_show=top_k,
        )
        print(f"  Згенеровано {len(data['steps'])} токенів: \"{data['answer']}\"")

        print("  [GEN-1] Generation Stream...")
        self.plot_generation_stream(data)

        print("  [GEN-2] Step by Step...")
        self.plot_step_by_step(data)

        print("  [GEN-3] Generation Heatmap...")
        self.plot_generation_heatmap(data)

        return data
