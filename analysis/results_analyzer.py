"""
Results Analyzer — аналіз результатів Задачі 3.

Будує:
1. ROC-крива і AUC для кожної метрики
2. Розподіли метрик: правильні vs неправильні відповіді
3. Confusion matrix по вердиктах
4. Кореляційна матриця метрик
5. Зведений звіт
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, classification_report,
                              confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from utils.plot_utils import save_or_show

METRICS = [
    "agreement_ratio",
    "mean_first_layer",
    "total_flips",
    "last_layer_ratio",
    "mean_final_confidence",
    "mean_logit_entropy",
    "mean_attention_entropy",
]

METRIC_LABELS = {
    "agreement_ratio":        "Agreement Ratio",
    "mean_first_layer":       "Mean First Layer",
    "total_flips":            "Total Flips",
    "last_layer_ratio":       "Last Layer Ratio",
    "mean_final_confidence":  "Final Confidence",
    "mean_logit_entropy":     "Logit Entropy",
    "mean_attention_entropy": "Attention Entropy",
}

# Для яких метрик вище = краще (True) / нижче = краще (False)
METRIC_DIRECTION = {
    "agreement_ratio":        True,
    "mean_first_layer":       False,
    "total_flips":            False,
    "last_layer_ratio":       True,
    "mean_final_confidence":  True,
    "mean_logit_entropy":     False,
    "mean_attention_entropy": False,
}


class ResultsAnalyzer:
    def __init__(self, config):
        self.cfg = config

    def _extract_arrays(self, results) -> dict:
        """Витягує numpy масиви з результатів."""
        data = {m: np.array([getattr(r, m) for r in results]) for m in METRICS}
        data["is_correct"] = np.array([r.is_correct for r in results])
        data["verdict"]    = [r.verdict    for r in results]
        data["category"]   = [r.category   for r in results]
        return data

    # ── 1. ROC криві ─────────────────────────────────────────────────
    def plot_roc_curves(self, results):
        """ROC-крива для кожної метрики як предиктора правильності."""
        data       = self._extract_arrays(results)
        y_true     = data["is_correct"]
        n_metrics  = len(METRICS)

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        auc_scores = {}

        for i, metric in enumerate(METRICS):
            ax  = axes[i]
            vals = data[metric]

            # Інвертуємо якщо нижче = краще
            if not METRIC_DIRECTION[metric]:
                vals = -vals

            fpr, tpr, _ = roc_curve(y_true, vals)
            roc_auc     = auc(fpr, tpr)
            auc_scores[metric] = roc_auc

            color = "green" if roc_auc >= 0.65 else \
                    "orange" if roc_auc >= 0.55 else "red"

            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
            ax.fill_between(fpr, tpr, alpha=0.1, color=color)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("False Positive Rate", fontsize=8)
            ax.set_ylabel("True Positive Rate", fontsize=8)
            ax.set_title(METRIC_LABELS[metric], fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)

        # Остання панель — порівняння AUC
        ax = axes[n_metrics]
        sorted_metrics = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        names  = [METRIC_LABELS[m] for m, _ in sorted_metrics]
        values = [v for _, v in sorted_metrics]
        colors = ["green" if v >= 0.65 else "orange" if v >= 0.55 else "red"
                  for v in values]
        bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0.5,  color="black", linestyle="--", alpha=0.4, label="Random")
        ax.axvline(0.65, color="green", linestyle="--", alpha=0.4, label="Good (0.65)")
        ax.set_xlim(0.3, 1.0)
        ax.set_title("Порівняння AUC", fontsize=9)
        ax.legend(fontsize=7)
        for bar, v in zip(bars, values):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8)

        for j in range(n_metrics + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "ROC Криві — чи передбачають внутрішні метрики правильність відповіді?\n"
            f"n={len(results)} питань",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        save_or_show(fig, "task3_01_roc_curves.png")
        return auc_scores

    # ── 2. Розподіли метрик ───────────────────────────────────────────
    def plot_metric_distributions(self, results):
        """Розподіл кожної метрики: правильні vs неправильні."""
        data    = self._extract_arrays(results)
        y_true  = data["is_correct"]
        correct = y_true == 1
        wrong   = y_true == 0

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()

        for i, metric in enumerate(METRICS):
            ax   = axes[i]
            vals = data[metric]

            c_vals = vals[correct]
            w_vals = vals[wrong]

            ax.hist(c_vals, bins=15, alpha=0.6, color="green",
                    label=f"Правильні (n={correct.sum()})", density=True)
            ax.hist(w_vals, bins=15, alpha=0.6, color="red",
                    label=f"Неправильні (n={wrong.sum()})", density=True)

            if len(c_vals) > 0 and len(w_vals) > 0:
                ax.axvline(c_vals.mean(), color="green", linestyle="--", lw=1.5)
                ax.axvline(w_vals.mean(), color="red",   linestyle="--", lw=1.5)

            ax.set_title(METRIC_LABELS[metric], fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

            # t-test p-value
            from scipy import stats
            if len(c_vals) > 1 and len(w_vals) > 1:
                _, p = stats.ttest_ind(c_vals, w_vals)
                color = "green" if p < 0.05 else "gray"
                ax.text(0.98, 0.97, f"p={p:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        for j in range(len(METRICS), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "Розподіл метрик: правильні vs неправильні відповіді\n"
            "Зелена пунктирна = середнє правильних | Червона = середнє неправильних",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_or_show(fig, "task3_02_distributions.png")

    # ── 3. Verdict confusion ──────────────────────────────────────────
    def plot_verdict_analysis(self, results):
        """Аналіз вердиктів: accuracy по кожному вердикту."""
        data     = self._extract_arrays(results)
        verdicts = data["verdict"]
        correct  = data["is_correct"]

        verdict_types = ["CONSISTENT", "CONVERGED", "UNCERTAIN", "CHANGED_MIND"]
        verdict_acc   = {}
        verdict_count = {}

        for v in verdict_types:
            mask = np.array([vi == v for vi in verdicts])
            if mask.sum() > 0:
                verdict_acc[v]   = correct[mask].mean()
                verdict_count[v] = mask.sum()
            else:
                verdict_acc[v]   = 0.0
                verdict_count[v] = 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy по вердиктах
        colors = {"CONSISTENT": "green", "CONVERGED": "blue",
                  "UNCERTAIN": "orange", "CHANGED_MIND": "red"}
        existing = [(v, verdict_acc[v], verdict_count[v])
                    for v in verdict_types if verdict_count[v] > 0]
        names  = [e[0] for e in existing]
        accs   = [e[1] for e in existing]
        counts = [e[2] for e in existing]
        bc     = [colors[n] for n in names]

        bars = ax1.bar(names, accs, color=bc, alpha=0.8)
        ax1.axhline(correct.mean(), color="black", linestyle="--",
                    alpha=0.5, label=f"Загальна accuracy ({correct.mean():.2f})")
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy по вердикту внутрішньої узгодженості")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        for bar, acc, cnt in zip(bars, accs, counts):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{acc:.2f}\n(n={cnt})", ha="center", fontsize=9)

        # Кількість по вердиктах
        ax2.pie(counts, labels=names, colors=bc, autopct="%1.0f%%",
                startangle=90, textprops={"fontsize": 9})
        ax2.set_title("Розподіл вердиктів")

        plt.suptitle(
            "Чи вердикт внутрішньої узгодженості корелює з правильністю?",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        save_or_show(fig, "task3_03_verdict_analysis.png")

    # ── 4. Кореляційна матриця ────────────────────────────────────────
    def plot_correlation_matrix(self, results):
        """Кореляція між метриками і правильністю відповіді."""
        data   = self._extract_arrays(results)
        keys   = METRICS + ["is_correct"]
        matrix = np.column_stack([data[k] for k in keys])

        corr = np.corrcoef(matrix.T)
        labels = [METRIC_LABELS.get(k, k) for k in keys[:-1]] + ["✓ Correct"]

        fig, ax = plt.subplots(figsize=(10, 9))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        sns.heatmap(
            corr,
            xticklabels=labels,
            yticklabels=labels,
            cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f",
            linewidths=0.3,
            ax=ax,
        )
        ax.set_title(
            "Кореляційна матриця метрик і правильності відповіді",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        plt.tight_layout()
        save_or_show(fig, "task3_04_correlation.png")

    # ── 5. Логістична регресія ────────────────────────────────────────
    def plot_logistic_regression(self, results):
        """
        Навчаємо логістичну регресію на всіх метриках
        і оцінюємо наскільки добре вони передбачають правильність.
        """
        data   = self._extract_arrays(results)
        X      = np.column_stack([data[m] for m in METRICS])
        y      = data["is_correct"]

        if y.sum() < 3 or (1 - y).sum() < 3:
            print("  Недостатньо зразків для логістичної регресії")
            return

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)

        # Cross-validation
        cv_scores = cross_val_score(clf, X_sc, y, cv=min(5, len(y)//5),
                                    scoring="roc_auc")

        # Fit для коефіцієнтів
        clf.fit(X_sc, y)
        coefs = clf.coef_[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Коефіцієнти
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        names  = [METRIC_LABELS[METRICS[i]] for i in sorted_idx]
        values = coefs[sorted_idx]
        colors = ["green" if v > 0 else "red" for v in values]

        bars = ax1.barh(range(len(names)), values, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.axvline(0, color="black", lw=0.8)
        ax1.set_title("Коефіцієнти логістичної регресії\n"
                      "(зелений = збільшує шанс правильної відповіді)")
        ax1.grid(axis="x", alpha=0.2)

        # CV scores
        ax2.bar(range(len(cv_scores)), cv_scores,
                color=["green" if s >= 0.65 else "orange" if s >= 0.55 else "red"
                       for s in cv_scores])
        ax2.axhline(0.5,  color="black",  linestyle="--", alpha=0.4, label="Random")
        ax2.axhline(0.65, color="green",  linestyle="--", alpha=0.5, label="Good")
        ax2.axhline(cv_scores.mean(), color="blue", linestyle="-",
                    label=f"Mean AUC = {cv_scores.mean():.3f}")
        ax2.set_ylim(0, 1.1)
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("ROC-AUC")
        ax2.set_title(f"Cross-validation ROC-AUC\nMean = {cv_scores.mean():.3f} "
                      f"± {cv_scores.std():.3f}")
        ax2.legend()
        ax2.grid(alpha=0.2)

        plt.suptitle(
            "Логістична Регресія: внутрішні метрики → правильність відповіді",
            fontsize=self.cfg.FONT_SIZE_TITLE,
        )
        plt.tight_layout()
        save_or_show(fig, "task3_05_logistic_regression.png")

    # ── 6. Зведений звіт ─────────────────────────────────────────────
    def print_summary(self, results, auc_scores: dict):
        """Виводить зведений звіт у консоль."""
        data    = self._extract_arrays(results)
        correct = data["is_correct"]
        n       = len(results)

        BOLD  = "\033[1m"
        GREEN = "\033[92m"
        CYAN  = "\033[96m"
        RESET = "\033[0m"

        print(f"\n{BOLD}{'═'*65}{RESET}")
        print(f"{BOLD}{CYAN}  РЕЗУЛЬТАТИ ЗАДАЧІ 3 — Зведений Звіт{RESET}")
        print(f"{BOLD}{'═'*65}{RESET}")
        print(f"  Датасет : {n} питань")
        print(f"  Accuracy: {correct.sum()}/{n} = {correct.mean():.1%}")

        print(f"\n  {BOLD}AUC по метриках (чи передбачають правильність):{RESET}")
        sorted_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        for metric, auc_val in sorted_auc:
            color = GREEN if auc_val >= 0.65 else "\033[93m" if auc_val >= 0.55 else "\033[91m"
            bar   = "█" * int(auc_val * 20) + "░" * (20 - int(auc_val * 20))
            print(f"  {color}{bar}{RESET} {auc_val:.3f}  {METRIC_LABELS[metric]}")

        best_metric, best_auc = sorted_auc[0]
        print(f"\n  {BOLD}Найкращий предиктор: "
              f"{GREEN}{METRIC_LABELS[best_metric]}{RESET}"
              f"{BOLD} (AUC={best_auc:.3f}){RESET}")

        # Середні значення по групах
        print(f"\n  {BOLD}Середні значення метрик:{RESET}")
        print(f"  {'Метрика':<28} {'Правильні':>10} {'Неправильні':>12} {'Різниця':>9}")
        print(f"  {'─'*60}")
        for metric in METRICS:
            vals = data[metric]
            c_mean = vals[correct == 1].mean() if (correct == 1).sum() > 0 else 0
            w_mean = vals[correct == 0].mean() if (correct == 0).sum() > 0 else 0
            diff   = c_mean - w_mean
            direction = "↑" if METRIC_DIRECTION[metric] else "↓"
            color = GREEN if (diff > 0) == METRIC_DIRECTION[metric] else "\033[91m"
            print(f"  {METRIC_LABELS[metric]:<28} "
                  f"{c_mean:>10.3f} {w_mean:>12.3f} "
                  f"{color}{diff:>+9.3f} {direction}{RESET}")

        # Висновок
        good_metrics = [m for m, a in auc_scores.items() if a >= 0.60]
        print(f"\n  {BOLD}Висновок:{RESET}")
        if good_metrics:
            names = ", ".join(METRIC_LABELS[m] for m in good_metrics)
            print(f"  {GREEN}✓ Метрики [{names}]{RESET}")
            print(f"    передбачають правильність відповіді краще за random (AUC≥0.60)")
            print(f"    → Гіпотеза Задачі 3 {GREEN}ПІДТВЕРДЖЕНА{RESET}")
        else:
            print(f"  Жодна метрика не досягла AUC≥0.60")
            print(f"  → Гіпотеза Задачі 3 не підтверджена на цьому датасеті")
            print(f"  → Потрібно більше даних або кращі метрики")

        print(f"{BOLD}{'═'*65}{RESET}")

    def run_full_analysis(self, results):
        """Запускає повний аналіз."""
        print("\n  Аналізуємо результати...")
        auc_scores = self.plot_roc_curves(results)
        self.plot_metric_distributions(results)
        self.plot_verdict_analysis(results)
        self.plot_correlation_matrix(results)
        self.plot_logistic_regression(results)
        self.print_summary(results, auc_scores)
        return auc_scores
