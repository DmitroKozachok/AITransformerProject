"""
MRI — "МРТ думки" GPT-2.

Показує як думка формується від шару до шару:

1. mri_thought_formation  — головний графік:
   - Зверху: теплова карта впевненості (шари × токени відповіді)
   - Знизу:  текст думки на кожному шарі (що модель "збиралась" сказати)
   - Права панель: увага на ключові токени питання

2. mri_attention_flow     — потік уваги від питання до відповіді
   Показує які слова питання "активували" кожне слово відповіді

3. mri_hidden_trajectory  — траєкторія hidden state у 2D (PCA)
   Як вектор "думки" рухався від шару до шару

4. mri_full_panel         — всі три панелі разом як один "знімок МРТ"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _get_parts(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    t = model.transformer
    return t.wte, t.wpe, t.drop, t.h, t.ln_f


def _collect_mri_data(model, tokenizer, question: str, answer: str) -> dict:
    """
    Один прохід через модель — збирає все потрібне для МРТ:
    - logit lens (впевненість і топ-1 на кожному шарі)
    - hidden states (для PCA)
    - attention (для потоку)
    """
    wte, wpe, drop, blocks, ln_f = _get_parts(model)
    num_layers = len(blocks)

    full_text   = question + answer
    inputs_q    = tokenizer(question,  return_tensors="pt")
    inputs_full = tokenizer(full_text, return_tensors="pt")

    q_len    = inputs_q["input_ids"].shape[1]
    full_ids = inputs_full["input_ids"]
    full_len = full_ids.shape[1]

    q_tokens = clean_tokens(tokenizer.convert_ids_to_tokens(full_ids[0, :q_len].tolist()))
    a_tokens = clean_tokens(tokenizer.convert_ids_to_tokens(full_ids[0, q_len:].tolist()))

    layer_top1_tokens = []   # (layers, ans_len) — думка кожного шару
    layer_top1_probs  = []   # (layers, ans_len) — впевненість
    hidden_states     = []   # (layers, full_len, emb_dim)
    attention_layers  = []   # (layers, heads, full_len, full_len)

    ans_positions = list(range(q_len - 1, full_len - 1))  # позиції що передбачають токени відповіді

    with torch.no_grad():
        pos_ids = torch.arange(full_len).unsqueeze(0)
        hidden  = wte(full_ids) + wpe(pos_ids)
        hidden  = drop(hidden)

        for block in blocks:
            # Увага
            attn_out, attn_w = block.attn(block.ln_1(hidden))[:2]
            hidden = hidden + attn_out
            hidden = hidden + block.mlp(block.ln_2(hidden))

            # Logit Lens
            normed = ln_f(hidden)
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(normed)[0]
            else:
                logits = normed[0] @ wte.weight.T

            probs    = torch.softmax(logits, dim=-1)
            top1_ids = probs.argmax(dim=-1)[ans_positions].tolist()
            top1_p   = probs.max(dim=-1).values[ans_positions].tolist()

            thought = clean_tokens(tokenizer.convert_ids_to_tokens(top1_ids))
            layer_top1_tokens.append(thought)
            layer_top1_probs.append(top1_p)
            hidden_states.append(hidden[0].numpy())
            attention_layers.append(attn_w[0].numpy())   # (heads, full, full)

    # Фінальна відповідь
    final_ids   = full_ids[0, q_len:].tolist()
    final_toks  = clean_tokens(tokenizer.convert_ids_to_tokens(final_ids))

    return {
        "question":         question,
        "answer":           answer.strip(),
        "q_tokens":         q_tokens,
        "a_tokens":         a_tokens,
        "final_tokens":     final_toks,
        "q_len":            q_len,
        "full_len":         full_len,
        "num_layers":       num_layers,
        "layer_thoughts":   layer_top1_tokens,  # (layers, ans_len)
        "layer_probs":      np.array(layer_top1_probs),  # (layers, ans_len)
        "hidden_states":    hidden_states,       # list of (full_len, emb)
        "attention_layers": attention_layers,    # list of (heads, full, full)
    }


class MRIVisualizer:
    def __init__(self, config):
        self.cfg = config

    # ── 1. Головний МРТ графік ────────────────────────────────────────
    def plot_thought_formation(self, data: dict):
        """
        Головний графік: як думка формується шар за шаром.

        Верхня панель: теплова карта впевненості (шари × токени відповіді)
        Нижня панель:  текст думки кожного шару
        """
        probs      = data["layer_probs"]      # (layers, ans_len)
        thoughts   = data["layer_thoughts"]   # (layers, ans_len)
        final_toks = data["final_tokens"]
        num_layers = data["num_layers"]
        ans_len    = len(final_toks)

        # Матриця збігу: 1 якщо думка шару = фінальний токен
        match_matrix = np.zeros((num_layers, ans_len))
        for layer_idx in range(num_layers):
            for tok_idx in range(min(ans_len, len(thoughts[layer_idx]))):
                if thoughts[layer_idx][tok_idx] == final_toks[tok_idx]:
                    match_matrix[layer_idx, tok_idx] = 1.0

        fig = plt.figure(figsize=(max(14, ans_len * 1.2), 12))
        gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 3, 3], hspace=0.4)

        # ── Панель 1: впевненість ──
        ax1 = fig.add_subplot(gs[0])
        im  = ax1.imshow(probs, aspect="auto", cmap="YlOrRd",
                         vmin=0, vmax=1, interpolation="nearest")
        ax1.set_yticks(range(num_layers))
        ax1.set_yticklabels([f"L{i+1}" for i in range(num_layers)], fontsize=8)
        ax1.set_xticks(range(ans_len))
        ax1.set_xticklabels(final_toks, rotation=45, ha="right", fontsize=9)
        ax1.set_title("Впевненість думки по шарах (яскравіше = впевненіше)",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        plt.colorbar(im, ax=ax1, fraction=0.02)

        # ── Панель 2: збіг думки з відповіддю ──
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(match_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")
        ax2.set_yticks(range(num_layers))
        ax2.set_yticklabels([f"L{i+1}" for i in range(num_layers)], fontsize=8)
        ax2.set_xticks(range(ans_len))
        ax2.set_xticklabels(final_toks, rotation=45, ha="right", fontsize=9)

        # Текст думки в клітинках
        for layer_idx in range(num_layers):
            for tok_idx in range(min(ans_len, len(thoughts[layer_idx]))):
                thought_tok = thoughts[layer_idx][tok_idx][:5]
                color = "white" if match_matrix[layer_idx, tok_idx] > 0.5 else "black"
                ax2.text(tok_idx, layer_idx, thought_tok,
                         ha="center", va="center", fontsize=6, color=color)
        ax2.set_title("Думка кожного шару (зелений = збіглась з відповіддю)",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)

        # ── Панель 3: текст думки по ключових шарах ──
        ax3 = fig.add_subplot(gs[2])
        ax3.axis("off")

        checkpoints = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
        checkpoints = sorted(set(max(1, c) for c in checkpoints))

        rows = []
        rows.append(["Шар", "Думка моделі", "Збіг з відповіддю"])
        for layer_idx in checkpoints:
            thought_text = " ".join(thoughts[layer_idx - 1][:ans_len])
            final_text   = " ".join(final_toks[:ans_len])
            # Рахуємо збіг
            matches = sum(
                1 for a, b in zip(thoughts[layer_idx-1][:ans_len], final_toks[:ans_len])
                if a == b
            )
            pct = matches / max(ans_len, 1) * 100
            rows.append([f"L{layer_idx}", thought_text[:50], f"{pct:.0f}%"])

        table = ax3.table(
            cellText=rows[1:],
            colLabels=rows[0],
            cellLoc="left",
            loc="center",
            colWidths=[0.08, 0.72, 0.20],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Підсвічуємо рядок фінального шару
        for col in range(3):
            table[len(checkpoints), col].set_facecolor("#d5f5e3")

        ax3.set_title("Еволюція думки по ключових шарах",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)

        plt.suptitle(
            f"МРТ Думки GPT-2\n"
            f"Питання: \"{data['question']}\"\n"
            f"Відповідь: \"{data['answer'][:60]}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE, y=1.01,
        )
        plt.tight_layout()
        save_or_show(fig, "mri_01_thought_formation.png")

    # ── 2. Потік уваги ────────────────────────────────────────────────
    def plot_attention_flow(self, data: dict):
        """
        Показує які токени питання "активували" кожен токен відповіді.
        Усереднена увага по всіх шарах і головах.
        """
        q_tokens   = data["q_tokens"]
        a_tokens   = data["a_tokens"]
        q_len      = data["q_len"]
        num_layers = data["num_layers"]

        # Усереднена увага відповіді → питання
        avg_attn = np.zeros((len(a_tokens), len(q_tokens)))
        for layer_attn in data["attention_layers"]:   # (heads, full, full)
            heads_avg = layer_attn.mean(axis=0)       # (full, full)
            ans_to_q  = heads_avg[q_len:, :q_len]     # (ans_len, q_len)
            h = min(len(a_tokens), ans_to_q.shape[0])
            w = min(len(q_tokens), ans_to_q.shape[1])
            avg_attn[:h, :w] += ans_to_q[:h, :w]
        avg_attn /= num_layers

        # Нормалізуємо рядки
        row_sum  = avg_attn.sum(axis=1, keepdims=True)
        avg_attn = avg_attn / (row_sum + 1e-9)

        fig, ax = plt.subplots(figsize=(max(10, len(q_tokens) * 0.9),
                                        max(6,  len(a_tokens) * 0.6)))
        sns.heatmap(
            avg_attn,
            xticklabels=q_tokens,
            yticklabels=a_tokens[:avg_attn.shape[0]],
            cmap="Blues",
            vmin=0, vmax=avg_attn.max(),
            annot=True, fmt=".2f",
            linewidths=0.3,
            ax=ax,
        )
        ax.set_xlabel("Токени питання (джерело уваги)")
        ax.set_ylabel("Токени відповіді (куди йде увага)")
        ax.set_title(
            "МРТ Потік Уваги: які слова питання активували кожне слово відповіді\n"
            f"Q: \"{data['question']}\"",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_or_show(fig, "mri_02_attention_flow.png")

    # ── 3. Траєкторія думки у просторі ───────────────────────────────
    def plot_hidden_trajectory(self, data: dict):
        """
        PCA hidden states токенів відповіді по шарах.
        Показує як вектор "думки" рухається у 2D просторі.
        """
        hidden_states = data["hidden_states"]   # list (layers) of (full_len, emb)
        q_len         = data["q_len"]
        a_tokens      = data["a_tokens"]
        num_layers    = data["num_layers"]

        # Збираємо hidden states тільки токенів відповіді
        ans_hidden = []   # (layers, ans_len, emb)
        for h in hidden_states:
            ans_h = h[q_len:q_len + len(a_tokens)]
            ans_hidden.append(ans_h)
        ans_hidden = np.array(ans_hidden)  # (layers, ans_len, emb)

        # PCA по всіх шарах разом
        layers, ans_len, emb = ans_hidden.shape
        flat = ans_hidden.reshape(-1, emb)
        pca  = PCA(n_components=2)
        coords = pca.fit_transform(flat).reshape(layers, ans_len, 2)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ── Ліво: траєкторія кожного токена по шарах ──
        ax = axes[0]
        cmap_layers = plt.cm.get_cmap("plasma", num_layers)
        cmap_tokens = plt.cm.get_cmap("tab10",  ans_len)

        for tok_idx in range(min(ans_len, 8)):   # максимум 8 токенів
            xs = coords[:, tok_idx, 0]
            ys = coords[:, tok_idx, 1]
            color = cmap_tokens(tok_idx)
            ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=1)
            # Стрілки напрямку
            for l in range(0, num_layers - 1, 3):
                dx, dy = xs[l+1] - xs[l], ys[l+1] - ys[l]
                ax.annotate("", xy=(xs[l+1], ys[l+1]), xytext=(xs[l], ys[l]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1))
            ax.scatter(xs[0],  ys[0],  color=color, s=80,  marker="o", zorder=5)
            ax.scatter(xs[-1], ys[-1], color=color, s=120, marker="*", zorder=6)
            label = a_tokens[tok_idx] if tok_idx < len(a_tokens) else ""
            ax.annotate(label, (xs[-1], ys[-1]),
                        fontsize=9, xytext=(5, 5), textcoords="offset points", color=color)

        ax.set_title("Траєкторія hidden state токенів по шарах\n(○=шар1, ★=шар12)",
                     fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        ax.grid(alpha=0.2)

        # ── Право: стан думки на кожному шарі (всі токени) ──
        ax2 = axes[1]
        for layer_idx in range(num_layers):
            color = cmap_layers(layer_idx)
            x = coords[layer_idx, :min(ans_len, 8), 0]
            y = coords[layer_idx, :min(ans_len, 8), 1]
            ax2.scatter(x, y, color=color, s=60, alpha=0.7, zorder=3)
            if layer_idx in [0, num_layers//2, num_layers-1]:
                for j, (xi, yi) in enumerate(zip(x, y)):
                    tok = a_tokens[j] if j < len(a_tokens) else ""
                    ax2.annotate(f"L{layer_idx+1}:{tok}", (xi, yi),
                                 fontsize=7, alpha=0.8)

        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(1, num_layers))
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, label="Шар")
        ax2.set_title("Стан думки на кожному шарі",
                      fontsize=self.cfg.FONT_SIZE_SUBTITLE)
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        ax2.grid(alpha=0.2)

        plt.suptitle(
            f"МРТ Траєкторія Думки у Просторі\nQ: \"{data['question']}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "mri_03_hidden_trajectory.png")

    # ── 4. Повний МРТ знімок ─────────────────────────────────────────
    def plot_full_mri(self, data: dict):
        """
        Один великий графік — всі аспекти думки разом.
        """
        probs      = data["layer_probs"]
        thoughts   = data["layer_thoughts"]
        final_toks = data["final_tokens"]
        q_tokens   = data["q_tokens"]
        a_tokens   = data["a_tokens"]
        q_len      = data["q_len"]
        num_layers = data["num_layers"]
        ans_len    = len(final_toks)

        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

        # ── A: Впевненість по шарах ──
        ax_conf = fig.add_subplot(gs[0, :2])
        im = ax_conf.imshow(probs, aspect="auto", cmap="YlOrRd",
                            vmin=0, vmax=1)
        ax_conf.set_yticks(range(num_layers))
        ax_conf.set_yticklabels([f"L{i+1}" for i in range(num_layers)], fontsize=7)
        ax_conf.set_xticks(range(ans_len))
        ax_conf.set_xticklabels(final_toks, rotation=45, ha="right", fontsize=8)
        ax_conf.set_title("A. Впевненість думки (шари × токени відповіді)",
                          fontsize=10)
        plt.colorbar(im, ax=ax_conf, fraction=0.015)

        # Підписуємо думку на ключових шарах
        for layer_idx in [0, num_layers//2-1, num_layers-1]:
            for tok_idx in range(min(ans_len, len(thoughts[layer_idx]))):
                t = thoughts[layer_idx][tok_idx][:4]
                p = probs[layer_idx, tok_idx]
                color = "white" if p > 0.5 else "gray"
                ax_conf.text(tok_idx, layer_idx, t,
                             ha="center", va="center", fontsize=6, color=color)

        # ── B: Потік уваги ──
        ax_attn = fig.add_subplot(gs[0, 2])
        avg_attn = np.zeros((len(a_tokens), len(q_tokens)))
        for layer_attn in data["attention_layers"]:
            heads_avg = layer_attn.mean(axis=0)
            h = min(len(a_tokens), heads_avg.shape[0] - q_len)
            w = min(len(q_tokens), q_len)
            avg_attn[:h, :w] += heads_avg[q_len:q_len+h, :w]
        avg_attn /= num_layers
        row_sum   = avg_attn.sum(axis=1, keepdims=True)
        avg_attn  = avg_attn / (row_sum + 1e-9)

        sns.heatmap(avg_attn, xticklabels=q_tokens,
                    yticklabels=a_tokens[:avg_attn.shape[0]],
                    cmap="Blues", ax=ax_attn, cbar=False)
        ax_attn.set_title("B. Потік уваги\nQ→A", fontsize=10)
        ax_attn.tick_params(axis="x", rotation=90, labelsize=7)
        ax_attn.tick_params(axis="y", labelsize=7)

        # ── C: Траєкторія PCA ──
        ax_pca = fig.add_subplot(gs[1, :2])
        hidden_states = data["hidden_states"]
        ans_hidden = np.array([h[q_len:q_len+len(a_tokens)] for h in hidden_states])
        layers, al, emb = ans_hidden.shape
        coords = PCA(n_components=2).fit_transform(
            ans_hidden.reshape(-1, emb)
        ).reshape(layers, al, 2)

        cmap_t = plt.cm.get_cmap("tab10", al)
        for tok_idx in range(min(al, 6)):
            xs, ys = coords[:, tok_idx, 0], coords[:, tok_idx, 1]
            color  = cmap_t(tok_idx)
            ax_pca.plot(xs, ys, "-o", color=color, alpha=0.6,
                        markersize=4, linewidth=1.5,
                        label=a_tokens[tok_idx] if tok_idx < len(a_tokens) else "")
            ax_pca.scatter(xs[-1], ys[-1], color=color, s=100, marker="*", zorder=5)
        ax_pca.legend(fontsize=8, loc="upper right")
        ax_pca.set_title("C. Траєкторія hidden state (PCA) — ○шар1 → ★шар12",
                         fontsize=10)
        ax_pca.grid(alpha=0.2)

        # ── D: Шкала формування думки ──
        ax_scale = fig.add_subplot(gs[1, 2])
        ax_scale.axis("off")

        checkpoints = sorted(set([1, num_layers//4, num_layers//2,
                                  num_layers*3//4, num_layers]))
        y_positions = np.linspace(0.95, 0.05, len(checkpoints))

        for y, layer_idx in zip(y_positions, checkpoints):
            thought_text = " ".join(thoughts[layer_idx-1][:min(ans_len, 6)])
            matches = sum(
                1 for a, b in zip(thoughts[layer_idx-1][:ans_len], final_toks[:ans_len])
                if a == b
            )
            pct   = matches / max(ans_len, 1)
            color = plt.cm.RdYlGn(pct)

            ax_scale.add_patch(mpatches.FancyBboxPatch(
                (0.0, y - 0.07), 1.0, 0.12,
                boxstyle="round,pad=0.01",
                facecolor=color, alpha=0.7, edgecolor="gray", linewidth=0.5,
            ))
            ax_scale.text(0.05, y, f"L{layer_idx:02d}", va="center",
                          fontsize=9, fontweight="bold")
            ax_scale.text(0.20, y, thought_text[:30], va="center", fontsize=8)
            ax_scale.text(0.92, y, f"{pct:.0%}", va="center",
                          fontsize=9, ha="right", fontweight="bold")

        ax_scale.set_xlim(0, 1)
        ax_scale.set_ylim(0, 1)
        ax_scale.set_title("D. Еволюція думки\n(зелений = збіглась)", fontsize=10)

        # Відповідь знизу
        ax_scale.add_patch(mpatches.FancyBboxPatch(
            (0.0, -0.05), 1.0, 0.08,
            boxstyle="round,pad=0.01",
            facecolor="#2ecc71", alpha=0.9, edgecolor="black", linewidth=1,
            transform=ax_scale.transAxes, clip_on=False,
        ))
        ax_scale.text(0.5, -0.01, f"✓ {' '.join(final_toks[:6])}",
                      va="center", ha="center", fontsize=9,
                      fontweight="bold", color="white",
                      transform=ax_scale.transAxes)

        plt.suptitle(
            f"МРТ Думки GPT-2\n"
            f"Питання: \"{data['question']}\"\n"
            f"Відповідь: \"{data['answer'][:60]}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        save_or_show(fig, "mri_04_full_panel.png")

    def plot_all(self, model, tokenizer, question: str, answer: str):
        """Збирає дані і будує всі 4 МРТ графіки."""
        print("  Збираємо дані МРТ...")
        data = _collect_mri_data(model, tokenizer, question, answer)

        print("  [MRI-1] Формування думки...")
        self.plot_thought_formation(data)

        print("  [MRI-2] Потік уваги...")
        self.plot_attention_flow(data)

        print("  [MRI-3] Траєкторія у просторі...")
        self.plot_hidden_trajectory(data)

        print("  [MRI-4] Повний МРТ знімок...")
        self.plot_full_mri(data)

        return data
