"""
Візуалізація User Satisfaction.

1. Score breakdown  — 5 метрик + weighted score
2. Gauge chart      — загальна оцінка як спідометр
3. Comparison grid  — порівняння кількох питань поруч
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.plot_utils import save_or_show

VERDICT_COLORS = {
    "SATISFACTORY":   "#2ecc71",
    "ACCEPTABLE":     "#f39c12",
    "UNSATISFACTORY": "#e74c3c",
}

METRIC_COLORS = {
    "coherence":   "#3498db",
    "relevance":   "#9b59b6",
    "confidence":  "#2ecc71",
    "consistency": "#e67e22",
    "fluency":     "#1abc9c",
}


class SatisfactionVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_score_breakdown(self, result: dict, suffix: str = ""):
        """Горизонтальні бари 5 метрик + weighted score."""
        scores  = result["scores"]
        verdict = result["verdict"]
        total   = result["weighted_score"]
        names   = list(scores.keys())
        vals    = [scores[n] for n in names]
        weights = [0.25, 0.20, 0.25, 0.20, 0.10]

        fig, (ax_main, ax_total) = plt.subplots(
            2, 1, figsize=(9, 6),
            gridspec_kw={"height_ratios": [5, 1.2]}
        )

        colors = [METRIC_COLORS.get(n, "gray") for n in names]
        bars   = ax_main.barh(names, vals, color=colors, alpha=0.85)
        ax_main.set_xlim(0, 1.15)
        ax_main.axvline(0.65, color="green",  linestyle="--", alpha=0.5, label="Satisfactory (0.65)")
        ax_main.axvline(0.40, color="orange", linestyle="--", alpha=0.5, label="Acceptable (0.40)")
        ax_main.legend(fontsize=8)
        ax_main.set_title(
            f"User Satisfaction — Score Breakdown\n"
            f"Q: \"{result['question'][:50]}\"",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        for bar, val, w in zip(bars, vals, weights):
            ax_main.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{val:.2f}  (w={w})", va="center", fontsize=9)
        ax_main.grid(axis="x", alpha=0.3)

        # Загальна оцінка
        vc = VERDICT_COLORS.get(verdict, "gray")
        ax_total.barh(["TOTAL"], [total], color=vc, alpha=0.9)
        ax_total.set_xlim(0, 1.15)
        ax_total.text(total + 0.01, 0, f"{total:.3f}  →  {verdict}",
                      va="center", fontsize=11, fontweight="bold", color=vc)
        ax_total.axvline(0.65, color="green",  linestyle="--", alpha=0.4)
        ax_total.axvline(0.40, color="orange", linestyle="--", alpha=0.4)
        ax_total.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        save_or_show(fig, f"satisfy_01_breakdown{suffix}.png")

    def plot_gauge(self, result: dict, suffix: str = ""):
        """Спідометр — загальна оцінка."""
        score   = result["weighted_score"]
        verdict = result["verdict"]
        vc      = VERDICT_COLORS.get(verdict, "gray")

        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"aspect": "equal"})
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.axis("off")

        # Фонова дуга (сіра)
        theta = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#ecf0f1", linewidth=18, solid_capstyle="round")

        # Кольорова дуга до score
        theta_score = np.linspace(np.pi, np.pi - score * np.pi, 200)
        ax.plot(np.cos(theta_score), np.sin(theta_score),
                color=vc, linewidth=18, solid_capstyle="round")

        # Зони
        for start, end, color, label in [
            (0.0, 0.4,  "#e74c3c", "Unsatisfactory"),
            (0.4, 0.65, "#f39c12", "Acceptable"),
            (0.65, 1.0, "#2ecc71", "Satisfactory"),
        ]:
            t = np.linspace(np.pi - start * np.pi, np.pi - end * np.pi, 50)
            mid_t = np.pi - (start + end) / 2 * np.pi
            ax.text(1.15 * np.cos(mid_t), 1.15 * np.sin(mid_t), label,
                    ha="center", va="center", fontsize=7, color=color)

        # Стрілка
        needle_angle = np.pi - score * np.pi
        ax.annotate("", xy=(0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.add_patch(plt.Circle((0, 0), 0.06, color="black", zorder=5))

        ax.text(0, -0.15, f"{score:.3f}", ha="center", fontsize=18,
                fontweight="bold", color=vc)
        ax.text(0, -0.28, verdict, ha="center", fontsize=11, color=vc)
        ax.set_title(f"User Satisfaction Gauge\n\"{result['answer'][:40]}\"",
                     fontsize=self.cfg.FONT_SIZE_SUBTITLE)

        plt.tight_layout()
        save_or_show(fig, f"satisfy_02_gauge{suffix}.png")

    def plot_thought_satisfaction_combined(self, thought_result: dict,
                                           sat_result: dict, suffix: str = ""):
        """
        Комбінований графік: Thought consistency (зліва) + Satisfaction (справа).
        Показує зв'язок між "думав те саме" і "задовільнить користувача".
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # ── 1. Thought agreement ratio по токенах ──
        per_token = thought_result.get("per_token", [])
        if per_token:
            tokens      = [t["final_token"] for t in per_token]
            agree       = [t["agreement_ratio"] for t in per_token]
            final_probs = [t["final_prob"]       for t in per_token]
            flips       = [t["flip_count"]       for t in per_token]

            x = np.arange(len(tokens))
            axes[0].bar(x, agree, color=[
                "#2ecc71" if a >= 0.6 else "#f39c12" if a >= 0.3 else "#e74c3c"
                for a in agree
            ], alpha=0.85)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            axes[0].set_ylim(0, 1.1)
            axes[0].axhline(0.6, color="green",  linestyle="--", alpha=0.5)
            axes[0].axhline(0.3, color="orange", linestyle="--", alpha=0.5)
            axes[0].set_title("Thought→Answer Agreement\n(частка шарів що вгадали)",
                              fontsize=9)
            axes[0].set_ylabel("Agreement Ratio")
            axes[0].grid(axis="y", alpha=0.3)

        # ── 2. Satisfaction метрики ──
        scores = sat_result["scores"]
        names  = list(scores.keys())
        vals   = [scores[n] for n in names]
        colors = [METRIC_COLORS.get(n, "gray") for n in names]
        axes[1].bar(names, vals, color=colors, alpha=0.85)
        axes[1].set_ylim(0, 1.1)
        axes[1].axhline(sat_result["weighted_score"], color="black",
                        linestyle="--", linewidth=1.5,
                        label=f"Weighted: {sat_result['weighted_score']:.2f}")
        axes[1].legend(fontsize=8)
        axes[1].set_title("Satisfaction Metrics", fontsize=9)
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", alpha=0.3)

        # ── 3. Radar: thought + satisfaction разом ──
        thought_overall = thought_result.get("overall", {})
        radar_labels = ["Agreement", "Confidence", "Coherence",
                        "Relevance", "Fluency", "Consistency"]
        radar_vals = [
            thought_overall.get("mean_agreement", 0),
            sat_result["scores"]["confidence"],
            sat_result["scores"]["coherence"],
            sat_result["scores"]["relevance"],
            sat_result["scores"]["fluency"],
            sat_result["scores"]["consistency"],
        ]

        n = len(radar_labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        radar_vals_closed = radar_vals + radar_vals[:1]

        ax3 = fig.add_subplot(133, polar=True)
        axes[2].set_visible(False)
        ax3.set_position(axes[2].get_position())

        ax3.plot(angles, radar_vals_closed, "o-", linewidth=2, color="#3498db")
        ax3.fill(angles, radar_vals_closed, alpha=0.2, color="#3498db")
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(radar_labels, size=8)
        ax3.set_ylim(0, 1)

        verdict_t = thought_result["overall"].get("verdict", "?")
        verdict_s = sat_result["verdict"]
        vc = VERDICT_COLORS.get(verdict_s, "gray")
        ax3.set_title(f"Combined Radar\nThought: {verdict_t} | Satisfy: {verdict_s}",
                      fontsize=9, color=vc, pad=15)

        plt.suptitle(
            f"Thought vs Answer + User Satisfaction\n"
            f"Q: \"{sat_result['question'][:50]}\"",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        save_or_show(fig, f"satisfy_03_combined{suffix}.png")

    def plot_all(self, thought_result: dict, sat_result: dict, suffix: str = ""):
        self.plot_score_breakdown(sat_result, suffix)
        self.plot_gauge(sat_result, suffix)
        self.plot_thought_satisfaction_combined(thought_result, sat_result, suffix)
