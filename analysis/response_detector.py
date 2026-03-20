"""
Response Detector — аналізує n-вимірний масив ознак і виносить вирок
про коректність/впевненість відповіді GPT-2.

Вирок по токену:
  CONFIDENT   — модель впевнена, патерни узгоджені
  UNCERTAIN   — невисока впевненість, висока ентропія
  SUSPICIOUS  — суперечливі сигнали (можлива галюцинація)

Вирок по відповіді в цілому:
  OK          — більшість токенів CONFIDENT
  UNCERTAIN   — значна частка UNCERTAIN
  HALLUCINATION_RISK — є SUSPICIOUS токени або загальна невпевненість
"""

import numpy as np


# ── Пороги (можна налаштувати в settings.py) ──────────────────────────
THRESHOLDS = {
    "logit_confidence_low":   0.30,   # нижче → невпевнено
    "logit_confidence_high":  0.70,   # вище  → впевнено
    "logit_entropy_high":     5.0,    # вище  → дуже розмитий розподіл
    "attention_entropy_high": 3.5,    # вище  → увага розсіяна
    "gradient_saliency_high": 0.80,   # вище  → токен "важливий" для loss
    "layer_confidence_std_high": 0.15,# вище  → нестабільне передбачення по шарах
}

LABEL_CONFIDENT  = "CONFIDENT"
LABEL_UNCERTAIN  = "UNCERTAIN"
LABEL_SUSPICIOUS = "SUSPICIOUS"


class ResponseDetector:
    def __init__(self, thresholds: dict = None):
        self.thr = {**THRESHOLDS, **(thresholds or {})}

    def classify_tokens(self, features: np.ndarray,
                        feature_names: list[str]) -> list[str]:
        """
        Класифікує кожен токен відповіді.
        features: (seq_len, NUM_FEATURES)
        Повертає список міток.
        """
        idx = {name: i for i, name in enumerate(feature_names)}
        labels = []

        for i in range(features.shape[0]):
            f = features[i]

            conf     = f[idx["logit_confidence"]]
            lent     = f[idx["logit_entropy"]]
            aent     = f[idx["attention_entropy"]]
            grad     = f[idx["gradient_saliency"]]
            std_conf = f[idx["layer_confidence_std"]]

            suspicious_flags = 0
            uncertain_flags  = 0

            # Впевненість логітів
            if conf < self.thr["logit_confidence_low"]:
                uncertain_flags += 1
            if lent > self.thr["logit_entropy_high"]:
                uncertain_flags += 1

            # Нестабільність по шарах — ознака галюцинації
            if std_conf > self.thr["layer_confidence_std_high"]:
                suspicious_flags += 1

            # Висока увага розсіяна — модель не знає на що спиратись
            if aent > self.thr["attention_entropy_high"]:
                uncertain_flags += 1

            # Дуже важливий для loss токен + низька впевненість = підозрілий
            if grad > self.thr["gradient_saliency_high"] and \
               conf < self.thr["logit_confidence_low"]:
                suspicious_flags += 1

            if suspicious_flags >= 1:
                labels.append(LABEL_SUSPICIOUS)
            elif uncertain_flags >= 2:
                labels.append(LABEL_UNCERTAIN)
            else:
                labels.append(LABEL_CONFIDENT)

        return labels

    def verdict(self, token_labels: list[str]) -> dict:
        """
        Зведений вирок для всієї відповіді.
        """
        total      = len(token_labels)
        if total == 0:
            return {"verdict": "UNKNOWN", "scores": {}}

        n_conf = token_labels.count(LABEL_CONFIDENT)
        n_unc  = token_labels.count(LABEL_UNCERTAIN)
        n_sus  = token_labels.count(LABEL_SUSPICIOUS)

        scores = {
            "confident_ratio":  round(n_conf / total, 3),
            "uncertain_ratio":  round(n_unc  / total, 3),
            "suspicious_ratio": round(n_sus  / total, 3),
        }

        if n_sus / total >= 0.2 or (n_sus > 0 and n_conf / total < 0.5):
            overall = "HALLUCINATION_RISK"
        elif n_unc / total >= 0.4:
            overall = "UNCERTAIN"
        else:
            overall = "OK"

        return {"verdict": overall, "scores": scores, "token_counts": {
            "CONFIDENT": n_conf, "UNCERTAIN": n_unc, "SUSPICIOUS": n_sus,
        }}

    def analyze(self, extracted: dict) -> dict:
        """
        Повний аналіз: приймає вивід FeatureExtractor.extract_for_response().
        Повертає dict з мітками, вердиктом і summary.
        """
        features      = extracted["features"]
        feature_names = extracted["feature_names"]
        tokens        = extracted["tokens"]

        token_labels  = self.classify_tokens(features, feature_names)
        result        = self.verdict(token_labels)

        result["tokens"]       = tokens
        result["token_labels"] = token_labels
        result["question"]     = extracted["question"]
        result["answer"]       = extracted["answer"]
        result["features"]     = features
        result["feature_names"]= feature_names

        return result

    def print_report(self, analysis: dict):
        """Виводить звіт у консоль."""
        print("\n" + "═" * 55)
        print("  DETECTOR REPORT")
        print("═" * 55)
        print(f"  Питання : {analysis['question']}")
        print(f"  Відповідь: {analysis['answer']}")
        print(f"\n  Вердикт: {analysis['verdict']}")
        print(f"  Scores : {analysis['scores']}")
        print(f"\n  По токенах:")
        for tok, lbl in zip(analysis["tokens"], analysis["token_labels"]):
            icon = {"CONFIDENT": "✓", "UNCERTAIN": "?", "SUSPICIOUS": "⚠"}.get(lbl, "·")
            print(f"    {icon} '{tok}' → {lbl}")
        print("═" * 55)