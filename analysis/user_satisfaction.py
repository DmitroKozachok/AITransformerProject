"""
User Satisfaction Estimator.

Оцінює чи відповідь GPT-2 задовольнить користувача за 5 метриками:

1. coherence_score     — зв'язність відповіді (перплексія)
2. relevance_score     — наскільки відповідь уважає токени питання
3. confidence_score    — середня впевненість logit по відповіді
4. consistency_score   — наскільки thought збігається з answer
5. fluency_score       — відсутність різких стрибків ентропії

Фінальна оцінка: weighted average → [0..1]
Вердикт: SATISFACTORY / ACCEPTABLE / UNSATISFACTORY
"""

import torch
import numpy as np
from scipy.stats import entropy as scipy_entropy
from utils.token_utils import clean_tokens


def _get_parts(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    t = model.transformer
    return t.wte, t.wpe, t.drop, t.h, t.ln_f


class UserSatisfactionEstimator:

    WEIGHTS = {
        "coherence":   0.25,
        "relevance":   0.20,
        "confidence":  0.25,
        "consistency": 0.20,
        "fluency":     0.10,
    }

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer

    # ── Метрики ───────────────────────────────────────────────────────

    def _coherence(self, answer: str) -> float:
        """
        Зв'язність через інверсну перплексію відповіді.
        Низька перплексія = модель "знає" ці слова разом = зв'язно.
        """
        inputs = self.tokenizer(answer, return_tensors="pt")
        ids    = inputs["input_ids"]
        if ids.shape[1] < 2:
            return 0.5

        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)

        with torch.no_grad():
            pos_ids = torch.arange(ids.shape[1]).unsqueeze(0)
            hidden  = wte(ids) + wpe(pos_ids)
            hidden  = drop(hidden)
            for block in blocks:
                hidden = block(hidden)[0]
            hidden = ln_f(hidden)
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(hidden)[0]
            else:
                logits = hidden[0] @ wte.weight.T

        loss = torch.nn.functional.cross_entropy(
            logits[:-1], ids[0, 1:]
        ).item()
        perplexity = np.exp(loss)
        # Нормалізуємо: perplexity 1=ідеально, 1000+=погано
        score = 1.0 / (1.0 + np.log1p(perplexity) / 10)
        return float(np.clip(score, 0, 1))

    def _relevance(self, question: str, answer: str) -> float:
        """
        Релевантність: скільки уваги токени відповіді приділяють токенам питання.
        """
        full_text  = question + answer
        inputs_q   = self.tokenizer(question,  return_tensors="pt")
        inputs_full= self.tokenizer(full_text, return_tensors="pt")
        q_len      = inputs_q["input_ids"].shape[1]
        full_ids   = inputs_full["input_ids"]
        full_len   = full_ids.shape[1]
        ans_len    = full_len - q_len

        if ans_len <= 0:
            return 0.5

        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)

        attn_to_q_per_layer = []
        with torch.no_grad():
            pos_ids = torch.arange(full_len).unsqueeze(0)
            hidden  = wte(full_ids) + wpe(pos_ids)
            hidden  = drop(hidden)
            for block in blocks:
                attn_out, attn_w = block.attn(block.ln_f(hidden) if hasattr(block, 'ln_f')
                                               else block.attn(block.ln_1(hidden)))[:2] \
                    if False else block.attn(block.ln_1(hidden))[:2]
                # attn_w: (1, heads, seq, seq)
                avg_attn = attn_w[0].mean(dim=0).numpy()   # (seq, seq)
                # Увага токенів відповіді до токенів питання
                ans_to_q = avg_attn[q_len:, :q_len].mean()
                attn_to_q_per_layer.append(float(ans_to_q))
                hidden = hidden + attn_out
                hidden = hidden + block.mlp(block.ln_2(hidden))

        return float(np.clip(np.mean(attn_to_q_per_layer) * 5, 0, 1))

    def _confidence(self, features: np.ndarray, feature_names: list) -> float:
        """Середня logit_confidence по токенах відповіді."""
        idx = feature_names.index("logit_confidence")
        return float(np.clip(features[:, idx].mean(), 0, 1))

    def _consistency(self, thought_result: dict) -> float:
        """Consistency з ThoughtVsAnswerAnalyzer."""
        overall = thought_result.get("overall", {})
        return float(overall.get("mean_agreement", 0.5))

    def _fluency(self, features: np.ndarray, feature_names: list) -> float:
        """
        Fluency: відсутність різких стрибків logit_entropy між сусідніми токенами.
        """
        idx  = feature_names.index("logit_entropy")
        ents = features[:, idx]
        if len(ents) < 2:
            return 0.5
        jumps = np.abs(np.diff(ents))
        # Менше стрибків = плавніше = краще
        score = 1.0 / (1.0 + jumps.mean())
        return float(np.clip(score, 0, 1))

    # ── Публічний API ─────────────────────────────────────────────────

    def estimate(self, question: str, answer: str,
                 features: np.ndarray, feature_names: list,
                 thought_result: dict) -> dict:
        """
        Повертає dict з усіма метриками і фінальним вердиктом.
        """
        scores = {
            "coherence":   self._coherence(answer),
            "relevance":   self._relevance(question, answer),
            "confidence":  self._confidence(features, feature_names),
            "consistency": self._consistency(thought_result),
            "fluency":     self._fluency(features, feature_names),
        }

        weighted = sum(scores[k] * self.WEIGHTS[k] for k in scores)

        if weighted >= 0.65:
            verdict = "SATISFACTORY"
        elif weighted >= 0.40:
            verdict = "ACCEPTABLE"
        else:
            verdict = "UNSATISFACTORY"

        return {
            "scores":          scores,
            "weighted_score":  round(float(weighted), 3),
            "verdict":         verdict,
            "question":        question,
            "answer":          answer,
        }

    def print_report(self, result: dict):
        print("\n" + "═" * 55)
        print("  USER SATISFACTION REPORT")
        print("═" * 55)
        print(f"  Питання : {result['question']}")
        print(f"  Відповідь: {result['answer']}")
        print(f"\n  Вердикт : {result['verdict']}")
        print(f"  Score   : {result['weighted_score']:.3f}")
        print(f"\n  По метриках:")
        for name, val in result["scores"].items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {name:<12} {bar} {val:.2f}")
        print("═" * 55)
