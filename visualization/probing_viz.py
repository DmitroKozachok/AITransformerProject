"""
Probing — що закодовано в hidden states.

1. Probing classifiers — чи можна з hidden state передбачити
   позицію токена, довжину слова, чи є це першим токеном слова
2. Causal tracing (спрощений) — що буде якщо "заморозити"
   активації одного шару при зміні вхідного тексту
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from utils.plot_utils import save_or_show
from utils.token_utils import clean_tokens


def _collect_all_hidden(model, tokenizer, texts: list[str]):
    """
    Збирає hidden states з усіх шарів для списку текстів.
    Повертає dict: {layer_idx: list of (seq, emb) arrays}
    """
    # Визначаємо кількість шарів через конфіг моделі
    if hasattr(model, 'h'):
        num_layers = len(model.h)
    else:
        num_layers = len(model.transformer.h)

    all_states = {i: [] for i in range(num_layers)}

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                output_hidden_states=True,
            )
        # hidden_states[0] = embedding, [1..n] = після кожного блоку
        for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):
            all_states[layer_idx].append(hidden[0].numpy())

    return all_states


class ProbingVisualizer:
    def __init__(self, config):
        self.cfg = config

    def plot_position_probing(self, model, tokenizer, texts: list[str]):
        """
        Probing: чи можна з hidden state кожного шару передбачити
        ПОЗИЦІЮ токена у реченні? (лінійний класифікатор)
        """
        all_states = _collect_all_hidden(model, tokenizer, texts)
        num_layers = len(all_states)

        X_per_layer = {i: [] for i in range(num_layers)}
        y = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            seq_len = inputs["input_ids"].shape[1]
            for pos in range(seq_len):
                y.append(min(pos, 4))

        y = np.array(y)

        for layer_idx in range(num_layers):
            for state in all_states[layer_idx]:
                for vec in state:
                    X_per_layer[layer_idx].append(vec)

        scores = []
        for layer_idx in range(num_layers):
            X = np.array(X_per_layer[layer_idx])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=500, random_state=42)
            cv_scores = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy")
            scores.append(cv_scores.mean())

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.get_cmap(self.cfg.CMAP)(np.linspace(0.3, 0.9, num_layers))
        bars = ax.bar([f"L{i+1}" for i in range(num_layers)], scores, color=colors)
        ax.axhline(1/5, color="red", linestyle="--", alpha=0.7, label="Random baseline (20%)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy (3-fold CV)")
        ax.set_title(
            "Probing: чи закодована позиція токена в hidden state?",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{s:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        save_or_show(fig, "24_position_probing.png")

    def plot_word_boundary_probing(self, model, tokenizer, texts: list[str]):
        """
        Probing: чи можна передбачити чи токен є початком слова?
        """
        all_states = _collect_all_hidden(model, tokenizer, texts)
        num_layers = len(all_states)

        X_per_layer = {i: [] for i in range(num_layers)}
        y = []

        for text in texts:
            inputs  = tokenizer(text, return_tensors="pt")
            raw_tok = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            clean   = clean_tokens(raw_tok)
            for tok in clean:
                y.append(1 if tok.startswith("·") else 0)

        y = np.array(y)

        for layer_idx in range(num_layers):
            for state in all_states[layer_idx]:
                for vec in state:
                    X_per_layer[layer_idx].append(vec)

        scores = []
        for layer_idx in range(num_layers):
            X = np.array(X_per_layer[layer_idx])
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf      = LogisticRegression(max_iter=500, random_state=42)
            cv_scores = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy")
            scores.append(cv_scores.mean())

        baseline = y.mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.get_cmap(self.cfg.CMAP)(np.linspace(0.3, 0.9, num_layers))
        bars = ax.bar([f"L{i+1}" for i in range(num_layers)], scores, color=colors)
        ax.axhline(max(baseline, 1-baseline), color="red", linestyle="--",
                   alpha=0.7, label=f"Majority baseline ({max(baseline, 1-baseline):.0%})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy (3-fold CV)")
        ax.set_title(
            "Probing: чи закодована межа слова в hidden state?",
            fontsize=self.cfg.FONT_SIZE_SUBTITLE,
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{s:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        save_or_show(fig, "25_word_boundary_probing.png")

    def plot_causal_tracing(self, model, tokenizer,
                            clean_text: str, corrupted_text: str):
        """
        Спрощений Causal Tracing: порівнює hidden states між двома реченнями.
        """
        def get_states(text):
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=32)
            tokens = clean_tokens(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    output_hidden_states=True,
                )
            # [1:] — пропускаємо embedding, беремо після кожного блоку
            states = [h[0].numpy() for h in outputs.hidden_states[1:]]
            return tokens, states

        tokens_c, states_c = get_states(clean_text)
        tokens_r, states_r = get_states(corrupted_text)

        min_len    = min(len(tokens_c), len(tokens_r))
        tokens     = tokens_c[:min_len]
        num_layers = len(states_c)

        diff_matrix = np.zeros((num_layers, min_len))
        for layer_idx in range(num_layers):
            sc = states_c[layer_idx][:min_len]
            sr = states_r[layer_idx][:min_len]
            diff_matrix[layer_idx] = np.linalg.norm(sc - sr, axis=-1)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            diff_matrix,
            xticklabels=tokens,
            yticklabels=[f"L{i+1}" for i in range(num_layers)],
            cmap="Reds",
            ax=ax,
        )
        ax.set_title(
            f"Causal Tracing — різниця hidden states між реченнями\n"
            f'Clean: "{clean_text[:40]}"\n'
            f'Corrupted: "{corrupted_text[:40]}"',
            fontsize=10,
        )
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        save_or_show(fig, "26_causal_tracing.png")

    def plot_probing_per_layer(self, model, tokenizer, texts: list[str]):
        """
        Для кожного шару окремо: стовпчиковий графік accuracy двох probing задач.
        """
        import os
        from utils.plot_utils import _output_dir

        all_states = _collect_all_hidden(model, tokenizer, texts)
        num_layers = len(all_states)
        out_dir    = os.path.join(_output_dir, "probing_per_layer")
        os.makedirs(out_dir, exist_ok=True)

        y_pos, y_bound = [], []
        for text in texts:
            inputs  = tokenizer(text, return_tensors="pt")
            raw_tok = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            clean   = clean_tokens(raw_tok)
            for pos, tok in enumerate(clean):
                y_pos.append(min(pos, 4))
                y_bound.append(1 if tok.startswith("·") else 0)
        y_pos   = np.array(y_pos)
        y_bound = np.array(y_bound)

        for layer_idx in range(num_layers):
            X = np.array([vec for state in all_states[layer_idx] for vec in state])
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf   = LogisticRegression(max_iter=500, random_state=42)
            s_pos = cross_val_score(clf, X_scaled, y_pos,   cv=3, scoring="accuracy").mean()
            s_bnd = cross_val_score(clf, X_scaled, y_bound, cv=3, scoring="accuracy").mean()

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["Позиція\n(5 класів)", "Межа слова\n(бінарна)"],
                          [s_pos, s_bnd],
                          color=["steelblue", "tomato"])
            ax.axhline(0.2, color="steelblue", linestyle="--", alpha=0.5,
                       label="baseline позиція (20%)")
            majority = max(y_bound.mean(), 1 - y_bound.mean())
            ax.axhline(majority, color="tomato", linestyle="--", alpha=0.5,
                       label=f"baseline межа ({majority:.0%})")
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Accuracy (3-fold CV)")
            ax.set_title(f"Probing — Шар {layer_idx+1}", fontsize=self.cfg.FONT_SIZE_SUBTITLE)
            ax.legend(fontsize=7)
            for bar, val in zip(bars, [s_pos, s_bnd]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.02, f"{val:.2f}",
                        ha="center", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            save_or_show(fig, f"probing_layer_{layer_idx+1:02d}.png", output_dir=out_dir)