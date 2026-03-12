"""
Збереження результатів аналізу: PNG + CSV + JSON у папку з датою/часом.
Структура:
  results/
    2024-01-15_14-32-10/
      plots/         ← всі PNG графіки
      attention_data.csv   ← усереднені матриці уваги
      metadata.json        ← параметри запуску + топ-зв'язки
"""

import os
import json
import numpy as np
from datetime import datetime


class ResultsSaver:
    def __init__(self, base_dir: str = "results"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir   = os.path.join(base_dir, timestamp)
        self.plots_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        print(f"\n📁 Результати зберігаються у: {self.run_dir}")

    @property
    def plots_path(self) -> str:
        return self.plots_dir

    def save_attention_csv(self, tokens, attentions):
        """
        Зберігає усереднені матриці уваги всіх шарів у CSV.
        Формат: layer, from_token, to_token, attention_score
        """
        import csv
        filepath = os.path.join(self.run_dir, "attention_data.csv")

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "from_token", "to_token", "attention_score"])

            for layer_idx, attn_tensor in enumerate(attentions):
                avg = attn_tensor[0].mean(dim=0).numpy()
                for r, from_tok in enumerate(tokens):
                    for c, to_tok in enumerate(tokens):
                        writer.writerow([
                            layer_idx + 1,
                            from_tok,
                            to_tok,
                            round(float(avg[r, c]), 6),
                        ])

        print(f"  ✓ CSV збережено: {filepath}")

    def save_metadata_json(self, config, tokens, attentions, top_k: int = 5):
        """
        Зберігає метадані запуску та топ-зв'язки уваги у JSON.
        """
        top_connections = []
        for layer_idx, attn_tensor in enumerate(attentions):
            avg = attn_tensor[0].mean(dim=0).numpy().copy()
            np.fill_diagonal(avg, 0)
            flat_idx = np.argsort(avg.flatten())[-top_k:][::-1]
            for idx in flat_idx:
                r, c = divmod(int(idx), len(tokens))
                top_connections.append({
                    "layer": layer_idx + 1,
                    "from": tokens[r],
                    "to":   tokens[c],
                    "score": round(float(avg[r, c]), 6),
                })

        metadata = {
            "timestamp":    datetime.now().isoformat(),
            "model":        config.MODEL_NAME,
            "text":         config.TEXT,
            "tokens":       tokens,
            "num_layers":   len(attentions),
            "inspect_layer": config.INSPECT_LAYER + 1,
            "top_connections": top_connections,
        }

        filepath = os.path.join(self.run_dir, "metadata.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"  ✓ JSON збережено: {filepath}")

    def save_all(self, config, tokens, attentions):
        """Зберігає CSV і JSON одним викликом."""
        print("\nЗберігаємо дані...")
        self.save_attention_csv(tokens, attentions)
        self.save_metadata_json(config, tokens, attentions)
        print(f"  ✓ Всі результати у: {self.run_dir}")