"""
Inner Speech — "внутрішній монолог" GPT-2.

Показує не просто топ-1 думку а повний розподіл:
що модель хотіла сказати, які варіанти розглядала,
наскільки була впевнена і коли "вирішила".

1. topk_candidates     — топ-K кандидатів на кожному кроці генерації
2. decision_moment     — в якому шарі модель "вирішила" що сказати
3. inner_monologue     — текст + альтернативи як "думки вголос"
4. probability_flow    — як змінювався розподіл ймовірностей по шарах
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _get_parts(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    t = model.transformer
    return t.wte, t.wpe, t.drop, t.h, t.ln_f


def _collect_inner_speech(model, tokenizer, question: str,
                           answer: str, top_k: int = 5) -> dict:
    """
    Для кожного токена відповіді і кожного шару збирає
    топ-K кандидатів з їх ймовірностями.
    """
    wte, wpe, drop, blocks, ln_f = _get_parts(model)
    num_layers = len(blocks)

    full_text   = question + answer
    inputs_q    = tokenizer(question,  return_tensors="pt")
    inputs_full = tokenizer(full_text, return_tensors="pt")

    q_len    = inputs_q["input_ids"].shape[1]
    full_ids = inputs_full["input_ids"]
    full_len = full_ids.shape[1]
    ans_len  = full_len - q_len

    q_tokens = clean_tokens(tokenizer.convert_ids_to_tokens(full_ids[0, :q_len].tolist()))
    a_tokens = clean_tokens(tokenizer.convert_ids_to_tokens(full_ids[0, q_len:].tolist()))

    # ans_positions[i] = позиція що передбачає i-й токен відповіді
    ans_positions = list(range(q_len - 1, full_len - 1))

    # per_token_per_layer[tok][layer] = [(token_str, prob), ...]
    per_token_per_layer = [[None] * num_layers for _ in range(ans_len)]

    with torch.no_grad():
        pos_ids = torch.arange(full_len).unsqueeze(0)
        hidden  = wte(full_ids) + wpe(pos_ids)
        hidden  = drop(hidden)

        for layer_idx, block in enumerate(blocks):
            hidden = block(hidden)[0]
            normed = ln_f(hidden)
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(normed)[0]
            else:
                logits = normed[0] @ wte.weight.T

            probs = torch.softmax(logits, dim=-1)

            for tok_idx, pos in enumerate(ans_positions):
                if tok_idx >= ans_len:
                    break
                p_vec    = probs[pos]
                topk_res = p_vec.topk(top_k)
                top_ids  = topk_res.indices.tolist()
                top_ps   = topk_res.values.tolist()
                top_toks = clean_tokens(tokenizer.convert_ids_to_tokens(top_ids))
                per_token_per_layer[tok_idx][layer_idx] = list(zip(top_toks, top_ps))

    return {
        "question":            question,
        "answer":              answer.strip(),
        "q_tokens":            q_tokens,
        "a_tokens":            a_tokens,
        "per_token_per_layer": per_token_per_layer,
        "num_layers":          num_layers,
        "ans_len":             ans_len,
        "top_k":               top_k,
    }


class InnerSpeechVisualizer:
    def __init__(self, config):
        self.cfg = config

    # ── 1. Топ-K кандидати ───────────────────────────────────────────
    def plot_topk_candidates(self, data: dict):
        """
        Для кожного токена відповіді — стовпчики топ-K кандидатів
        на фінальному шарі. Показує що модель "хотіла" сказати.
        """
        a_tokens   = data["a_tokens"]
        ppl        = data["per_token_per_layer"]
        ans_len    = data["ans_len"]
        top_k      = data["top_k"]
        num_layers = data["num_layers"]

        cols = min(ans_len, 8)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 2.2, 5))
        if cols == 1:
            axes = [axes]

        cmap = plt.cm.get_cmap(self.cfg.CMAP)

        for tok_idx, ax in enumerate(axes):
            if tok_idx >= ans_len:
                ax.set_visible(False)
                continue

            candidates = ppl[tok_idx][num_layers - 1]
            if candidates is None:
                ax.set_visible(False)
                continue

            toks  = [c[0] for c in candidates]
            probs = [c[1] for c in candidates]
            colors = [cmap(p) for p in probs]

            ax.barh(range(top_k), probs[::-1], color=colors[::-1])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(toks[::-1], fontsize=9)
            ax.set_xlim(0, 1)

            final_tok = a_tokens[tok_idx] if tok_idx < len(a_tokens) else ""
            title_color = "green" if toks[0] == final_tok else "red"
            ax.set_title(f"'{final_tok}'", fontsize=10, color=title_color)
            ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
            ax.grid(axis="x", alpha=0.3)
            for i, p in enumerate(probs[::-1]):
                ax.text(p + 0.01, i, f"{p:.2f}", va="center", fontsize=7)

        plt.suptitle(
            f"Топ-{top_k} кандидати що хотів сказати GPT-2\n"
            f"Q: \"{data['question']}\"\n"
            f"Зелений = вибрав топ-1 | Червоний = вибрав інший варіант",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.82)
        save_or_show(fig, "inner_01_topk_candidates.png")

    # ── 2. Момент рішення ────────────────────────────────────────────
    def plot_decision_moment(self, data: dict):
        """
        В якому шарі модель "вирішила" що сказати і наскільки стабільно.
        """
        a_tokens   = data["a_tokens"]
        ppl        = data["per_token_per_layer"]
        ans_len    = data["ans_len"]
        num_layers = data["num_layers"]

        decision_layers   = []
        stability_scores  = []
        confidence_curves = []

        for tok_idx in range(ans_len):
            final_tok = a_tokens[tok_idx] if tok_idx < len(a_tokens) else ""
            decided   = -1
            confs     = []
            top1_seq  = []

            for layer_idx in range(num_layers):
                candidates = ppl[tok_idx][layer_idx]
                if candidates is None:
                    confs.append(0)
                    top1_seq.append("")
                    continue
                top1_tok  = candidates[0][0]
                top1_prob = candidates[0][1]
                confs.append(top1_prob)
                top1_seq.append(top1_tok)

            for layer_idx, t in enumerate(top1_seq):
                if t == final_tok and decided == -1:
                    decided = layer_idx + 1

            # Стабільність: частка шарів де топ-1 не змінювався
            flips = sum(1 for i in range(1, len(top1_seq))
                        if top1_seq[i] != top1_seq[i-1])
            stability = 1.0 - flips / max(num_layers - 1, 1)

            decision_layers.append(decided if decided > 0 else num_layers)
            stability_scores.append(stability)
            confidence_curves.append(confs)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        display_toks = [a_tokens[i] if i < len(a_tokens) else f"t{i}"
                        for i in range(ans_len)]

        # Момент рішення
        colors = [plt.cm.RdYlGn(1 - d / num_layers) for d in decision_layers]
        bars = ax1.bar(display_toks, decision_layers, color=colors, alpha=0.85)
        ax1.set_ylabel("Шар")
        ax1.set_ylim(0, num_layers + 1)
        ax1.set_title("Момент рішення\n(нижче = раніше вирішив)", fontsize=10)
        ax1.axhline(num_layers // 2, color="orange", linestyle="--", alpha=0.5)
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, decision_layers):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1, f"L{val}", ha="center", fontsize=8)

        # Стабільність
        colors2 = [plt.cm.RdYlGn(s) for s in stability_scores]
        ax2.bar(display_toks, stability_scores, color=colors2, alpha=0.85)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Стабільність думки\n(вище = менше коливань)", fontsize=10)
        ax2.tick_params(axis="x", rotation=45)

        # Криві впевненості
        cmap_t = plt.cm.get_cmap("tab10", min(ans_len, 8))
        for tok_idx in range(min(ans_len, 8)):
            curve = confidence_curves[tok_idx]
            label = a_tokens[tok_idx] if tok_idx < len(a_tokens) else f"t{tok_idx}"
            ax3.plot(range(1, num_layers+1), curve,
                     color=cmap_t(tok_idx), label=f"'{label}'",
                     linewidth=1.8, alpha=0.85)
            dl = decision_layers[tok_idx] - 1
            if 0 <= dl < len(curve):
                ax3.scatter(dl+1, curve[dl], color=cmap_t(tok_idx), s=80, zorder=5)

        ax3.set_xlabel("Шар")
        ax3.set_ylabel("Впевненість топ-1")
        ax3.set_ylim(0, 1.05)
        ax3.set_title("Криві впевненості\n(● = момент рішення)", fontsize=10)
        ax3.legend(fontsize=7, bbox_to_anchor=(1.01, 1))
        ax3.grid(alpha=0.2)

        plt.suptitle(
            f"Момент Рішення GPT-2\nQ: \"{data['question']}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "inner_02_decision_moment.png")

    # ── 3. Внутрішній монолог ────────────────────────────────────────
    def plot_inner_monologue(self, data: dict):
        """
        Таблиця: що думала модель на кожному ключовому шарі.
        Для кожного токена — топ-3 варіанти і їх ймовірності.
        """
        a_tokens   = data["a_tokens"]
        ppl        = data["per_token_per_layer"]
        ans_len    = data["ans_len"]
        num_layers = data["num_layers"]

        checkpoints = sorted(set([
            1, num_layers//4, num_layers//2,
            num_layers*3//4, num_layers
        ]))
        display_cols = min(ans_len, 8)

        fig, ax = plt.subplots(
            figsize=(max(12, display_cols * 2), len(checkpoints) * 1.5 + 2)
        )
        ax.axis("off")

        col_labels = ["Шар"] + [
            f"'{a_tokens[i]}'" if i < len(a_tokens) else f"tok{i}"
            for i in range(display_cols)
        ]
        cell_text   = []
        cell_colors = []

        for layer_idx in checkpoints:
            row   = [f"L{layer_idx:02d}"]
            r_col = ["#f0f0f0"]

            for tok_idx in range(display_cols):
                candidates = ppl[tok_idx][layer_idx - 1]
                if candidates is None:
                    row.append("—")
                    r_col.append("white")
                    continue

                final_tok = a_tokens[tok_idx] if tok_idx < len(a_tokens) else ""
                lines = []
                for tok, prob in candidates[:3]:
                    marker = "▶" if tok == final_tok else " "
                    lines.append(f"{marker}{tok} {prob:.2f}")
                row.append("\n".join(lines))

                top1 = candidates[0][0]
                if top1 == final_tok:
                    r_col.append("#d5f5e3")
                elif final_tok in [c[0] for c in candidates]:
                    r_col.append("#fef9e7")
                else:
                    r_col.append("#fadbd8")

            cell_text.append(row)
            cell_colors.append(r_col)

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 3.5)

        legend_items = [
            mpatches.Patch(color="#d5f5e3", label="Топ-1 = обраний токен"),
            mpatches.Patch(color="#fef9e7", label="Обраний є в топ-3"),
            mpatches.Patch(color="#fadbd8", label="Обраного немає в топ-3"),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=8)

        plt.suptitle(
            f"Внутрішній Монолог GPT-2 — топ-3 кандидати по шарах\n"
            f"Q: \"{data['question']}\"\n"
            f"▶ = фінально обраний токен",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "inner_03_monologue.png")

    # ── 4. Потік ймовірностей ────────────────────────────────────────
    def plot_probability_flow(self, data: dict):
        """
        Для кожного токена відповіді — як змінювались ймовірності
        топ-K кандидатів від шару до шару.
        """
        a_tokens   = data["a_tokens"]
        ppl        = data["per_token_per_layer"]
        ans_len    = data["ans_len"]
        num_layers = data["num_layers"]

        cols = min(ans_len, 6)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 5))
        if cols == 1:
            axes = [axes]

        for tok_idx, ax in enumerate(axes):
            if tok_idx >= ans_len:
                ax.set_visible(False)
                continue

            final_tok = a_tokens[tok_idx] if tok_idx < len(a_tokens) else ""

            # Збираємо всіх унікальних кандидатів
            all_cands = {}
            for layer_idx in range(num_layers):
                cands = ppl[tok_idx][layer_idx]
                if cands:
                    for tok, prob in cands:
                        if tok not in all_cands:
                            all_cands[tok] = np.zeros(num_layers)
                        all_cands[tok][layer_idx] = prob

            # Топ-5 за максимальною ймовірністю
            sorted_cands = sorted(all_cands.items(),
                                  key=lambda x: x[1].max(), reverse=True)[:5]

            cmap = plt.cm.get_cmap("tab10", len(sorted_cands))
            layers_x = range(1, num_layers + 1)

            for i, (tok, probs) in enumerate(sorted_cands):
                is_final = tok == final_tok
                ax.plot(layers_x, probs,
                        color=cmap(i),
                        linewidth=2.5 if is_final else 1.2,
                        linestyle="-" if is_final else "--",
                        alpha=0.9 if is_final else 0.5,
                        label=f"{'▶ ' if is_final else ''}{tok}")
                ax.fill_between(layers_x, probs,
                                alpha=0.25 if is_final else 0.08,
                                color=cmap(i))

            ax.set_xlim(1, num_layers)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Шар", fontsize=8)
            if tok_idx == 0:
                ax.set_ylabel("Ймовірність", fontsize=8)

            title_color = "green" if sorted_cands and sorted_cands[0][0] == final_tok else "red"
            ax.set_title(f"'{final_tok}'", fontsize=10, color=title_color)
            ax.legend(fontsize=6, loc="upper left")
            ax.grid(alpha=0.2)

        plt.suptitle(
            f"Потік Ймовірностей по шарах\n"
            f"Q: \"{data['question']}\"  |  ▶ = обраний токен",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        save_or_show(fig, "inner_04_probability_flow.png")

    def plot_all(self, model, tokenizer, question: str, answer: str, top_k: int = 5):
        """Збирає дані і будує всі 4 Inner Speech графіки."""
        print("  Збираємо Inner Speech дані...")
        data = _collect_inner_speech(model, tokenizer, question, answer, top_k=top_k)

        print("  [INNER-1] Топ-K кандидати...")
        self.plot_topk_candidates(data)

        print("  [INNER-2] Момент рішення...")
        self.plot_decision_moment(data)

        print("  [INNER-3] Внутрішній монолог...")
        self.plot_inner_monologue(data)

        print("  [INNER-4] Потік ймовірностей...")
        self.plot_probability_flow(data)

        return data
