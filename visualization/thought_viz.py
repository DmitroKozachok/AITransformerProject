"""
Візуалізація Thought vs Answer.

1. Thought trajectory   — для кожного токена: що думала модель по шарах
2. Agreement heatmap    — шари × токени відповіді (збіг з фінальним)
3. Flip heatmap         — де і скільки разів модель "передумала"
4. Confidence evolution — як зростала/падала впевненість по шарах
5. Summary bar          — agreement_ratio і flip_count по токенах
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from utils.plot_utils import save_or_show

VERDICT_COLORS = {
    "CONSISTENT":   "#2ecc71",
    "CONVERGED":    "#3498db",
    "UNCERTAIN":    "#f39c12",
    "CHANGED_MIND": "#e74c3c",
    "UNKNOWN":      "#95a5a6",
}


class ThoughtVisualizer:
    def __init__(self, config):
        self.cfg = config

    # ── 1. Trajectory ─────────────────────────────────────────────────
    def plot_thought_trajectory(self, result: dict):
        """
        Для кожного токена відповіді — горизонтальна смужка:
        колір = який токен думала модель на кожному шарі.
        Фінальний токен позначений рамкою.
        """
        per_token  = result["per_token"]
        num_layers = result["num_layers"]
        if not per_token:
            return

        n_tokens = len(per_token)
        fig, axes = plt.subplots(n_tokens, 1,
                                  figsize=(max(12, num_layers * 0.8), n_tokens * 1.2 + 1))
        if n_tokens == 1:
            axes = [axes]

        for row, (ax, tok_data) in enumerate(zip(axes, per_token)):
            thought_toks = tok_data["thought_tokens"]
            thought_probs= np.array(tok_data["thought_probs"])
            final_tok    = tok_data["final_token"]

            # Унікальні токени → кольори
            unique = list(dict.fromkeys(thought_toks + [final_tok]))
            cmap   = plt.cm.get_cmap("tab20", len(unique))
            tok_color = {t: cmap(i) for i, t in enumerate(unique)}

            for layer_idx, (ttok, tprob) in enumerate(zip(thought_toks, thought_probs)):
                color = tok_color[ttok]
                ax.add_patch(plt.Rectangle(
                    (layer_idx, 0), 1, 1, color=color, alpha=0.7 + 0.3 * tprob
                ))
                ax.text(layer_idx + 0.5, 0.5, ttok[:5],
                        ha="center", va="center", fontsize=6,
                        color="white" if tprob > 0.4 else "black")

            # Рамка навколо останнього шару якщо збігся з фінальним
            lc = "green" if tok_data["final_was_thought"] else "red"
            ax.add_patch(plt.Rectangle(
                (num_layers - 1, 0), 1, 1,
                fill=False, edgecolor=lc, linewidth=2.5
            ))

            ax.set_xlim(0, num_layers)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.5])
            ax.set_yticklabels([f"'{final_tok}'"], fontsize=8)
            ax.set_xticks([])
            if row == 0:
                ax.set_title("Thought Trajectory (шари →, рамка = останній шар)",
                             fontsize=self.cfg.FONT_SIZE_SUBTITLE)

        # X-мітки тільки знизу
        axes[-1].set_xticks(np.arange(num_layers) + 0.5)
        axes[-1].set_xticklabels([f"L{i+1}" for i in range(num_layers)],
                                  rotation=45, ha="right", fontsize=7)

        plt.suptitle(
            f"Thought Trajectory  |  Вердикт: {result['overall'].get('verdict','?')}",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.05)
        save_or_show(fig, "thought_01_trajectory.png")

    # ── 2. Agreement heatmap ──────────────────────────────────────────
    def plot_agreement_heatmap(self, result: dict):
        """
        Heatmap: рядки = шари, стовпці = токени відповіді.
        Значення = 1 якщо думка шару збіглась з фінальним токеном, 0 — ні.
        """
        per_token  = result["per_token"]
        num_layers = result["num_layers"]
        if not per_token:
            return

        answer_tokens = [t["final_token"] for t in per_token]
        matrix = np.zeros((num_layers, len(per_token)))

        for col, tok_data in enumerate(per_token):
            final_tok    = tok_data["final_token"]
            thought_toks = tok_data["thought_tokens"]
            for row, tt in enumerate(thought_toks):
                matrix[row, col] = 1.0 if tt == final_tok else 0.0

        fig, ax = plt.subplots(figsize=(max(8, len(per_token) * 1.2), 7))
        sns.heatmap(
            matrix,
            xticklabels=answer_tokens,
            yticklabels=[f"L{i+1}" for i in range(num_layers)],
            cmap="RdYlGn",
            vmin=0, vmax=1,
            annot=True, fmt=".0f",
            linewidths=0.3,
            ax=ax,
        )
        ax.set_title(
            "Agreement Heatmap\n(1 = думка шару збіглась з фінальним токеном)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_or_show(fig, "thought_02_agreement_heatmap.png")

    # ── 3. Confidence evolution ───────────────────────────────────────
    def plot_confidence_evolution(self, result: dict):
        """
        Для кожного токена відповіді — лінія впевненості по шарах.
        """
        per_token  = result["per_token"]
        num_layers = result["num_layers"]
        if not per_token:
            return

        fig, ax = plt.subplots(figsize=(11, 5))
        cmap = plt.cm.get_cmap("tab10", len(per_token))

        for i, tok_data in enumerate(per_token):
            probs = tok_data["thought_probs"]
            label = f"'{tok_data['final_token']}'"
            color = cmap(i)
            ax.plot(range(1, num_layers + 1), probs,
                    marker="o", linewidth=1.8, color=color, label=label, alpha=0.85)
            # Позначаємо де вперше "вгадало"
            fa = tok_data["first_agreement_layer"]
            if 0 < fa <= num_layers:
                ax.scatter(fa, probs[fa - 1], s=100, color=color,
                           zorder=5, marker="*")

        ax.set_xlabel("Шар")
        ax.set_ylabel("Впевненість (max softmax prob)")
        ax.set_xticks(range(1, num_layers + 1))
        ax.set_xticklabels([f"L{i}" for i in range(1, num_layers + 1)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.set_title(
            "Confidence Evolution по шарах  (★ = перший шар що вгадав)",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.grid(alpha=0.25)
        plt.tight_layout()
        save_or_show(fig, "thought_03_confidence_evolution.png")

    # ── 4. Summary bar ────────────────────────────────────────────────
    def plot_summary(self, result: dict):
        """
        Стовпчикові графіки: agreement_ratio і flip_count по токенах.
        """
        per_token = result["per_token"]
        if not per_token:
            return

        tokens      = [t["final_token"] for t in per_token]
        agree_ratios= [t["agreement_ratio"]       for t in per_token]
        flips       = [t["flip_count"]             for t in per_token]
        final_probs = [t["final_prob"]             for t in per_token]
        first_layers= [t["first_agreement_layer"]  for t in per_token]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        verdict = result["overall"].get("verdict", "?")
        vc = VERDICT_COLORS.get(verdict, "gray")

        # Agreement ratio
        bars = axes[0, 0].bar(tokens, agree_ratios,
                              color=[("green" if v >= 0.6 else "orange" if v >= 0.3 else "red")
                                     for v in agree_ratios])
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].set_title("Agreement Ratio\n(частка шарів що вгадали фінальний токен)")
        axes[0, 0].axhline(0.6, color="green",  linestyle="--", alpha=0.5)
        axes[0, 0].axhline(0.3, color="orange", linestyle="--", alpha=0.5)
        axes[0, 0].tick_params(axis="x", rotation=45)
        for bar, v in zip(bars, agree_ratios):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.02, f"{v:.2f}", ha="center", fontsize=8)

        # Flip count
        axes[0, 1].bar(tokens, flips, color="steelblue", alpha=0.8)
        axes[0, 1].set_title("Flip Count\n(кількість змін думки між шарами)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Final confidence
        axes[1, 0].bar(tokens, final_probs,
                       color=[("green" if v >= 0.5 else "tomato") for v in final_probs])
        axes[1, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].set_title("Final Confidence\n(впевненість у фінальному токені)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # First agreement layer
        valid_fa = [l for l in first_layers if l > 0]
        axes[1, 1].bar(
            [t for t, l in zip(tokens, first_layers) if l > 0],
            valid_fa,
            color="mediumpurple", alpha=0.8,
        )
        axes[1, 1].set_title("First Agreement Layer\n(з якого шару модель 'знала' відповідь)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].set_ylabel("Шар")

        plt.suptitle(
            f"Thought vs Answer Summary  |  Вердикт: {verdict}  "
            f"  mean_agree={result['overall'].get('mean_agreement','?')}",
            fontsize=self.cfg.FONT_SIZE_TITLE,
            color=vc,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        save_or_show(fig, "thought_04_summary.png")

    def plot_all(self, result: dict):
        self.plot_thought_trajectory(result)
        self.plot_agreement_heatmap(result)
        self.plot_confidence_evolution(result)
        self.plot_summary(result)