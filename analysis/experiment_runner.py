"""
Experiment Runner — Задача 3.

Запускає систему на датасеті і збирає:
  - метрики внутрішньої узгодженості (agreement_ratio, flip_count, ...)
  - правильність відповіді (0/1)

Зберігає результати у CSV для подальшого аналізу.
"""

import os
import csv
import time
import numpy as np
import torch
from dataclasses import dataclass, asdict, fields

from analysis.dataset_loader import check_answer
from utils.token_utils import clean_tokens


@dataclass
class SampleResult:
    # Ідентифікація
    question:         str
    correct_answer:   str
    model_answer:     str
    category:         str
    is_correct:       int    # 0 або 1

    # Метрики внутрішньої узгодженості
    agreement_ratio:      float  # частка шарів що вгадали фінальний токен
    mean_first_layer:     float  # середній шар першої згоди
    total_flips:          int    # сумарна кількість змін думки
    last_layer_ratio:     float  # частка токенів де останній шар вгадав
    mean_final_confidence:float  # середня впевненість у фінальних токенах
    mean_logit_entropy:   float  # середня ентропія logit розподілу
    mean_attention_entropy:float # середня ентропія уваги
    verdict:              str    # CONSISTENT / CONVERGED / UNCERTAIN / CHANGED_MIND

    # Час
    elapsed_sec: float


def _get_parts(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    t = model.transformer
    return t.wte, t.wpe, t.drop, t.h, t.ln_f


class ExperimentRunner:
    def __init__(self, model, tokenizer, output_dir: str):
        self.model      = model
        self.tokenizer  = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _generate_answer(self, question: str, max_new_tokens: int = 15) -> str:
        """Генерує відповідь GPT-2 на питання."""
        inputs   = self.tokenizer(question, return_tensors="pt")
        eos_id   = self.tokenizer.eos_token_id

        with torch.no_grad():
            out = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
            )
        answer_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def _compute_metrics(self, question: str, answer: str) -> dict:
        """
        Обчислює всі метрики внутрішньої узгодженості для пари питання-відповідь.
        """
        from scipy.stats import entropy as scipy_entropy

        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)
        num_layers = len(blocks)

        full_text   = question + answer
        inputs_q    = self.tokenizer(question,  return_tensors="pt")
        inputs_full = self.tokenizer(full_text, return_tensors="pt")

        q_len    = inputs_q["input_ids"].shape[1]
        full_ids = inputs_full["input_ids"]
        full_len = full_ids.shape[1]
        ans_len  = full_len - q_len

        if ans_len <= 0:
            return self._empty_metrics(num_layers)

        a_tokens = clean_tokens(
            self.tokenizer.convert_ids_to_tokens(full_ids[0, q_len:].tolist())
        )
        ans_positions = list(range(q_len - 1, full_len - 1))

        # Logit Lens
        layer_top1_ids   = []
        layer_top1_probs = []
        layer_entropies  = []
        attn_entropies   = []

        with torch.no_grad():
            pos_ids = torch.arange(full_len).unsqueeze(0)
            hidden  = wte(full_ids) + wpe(pos_ids)
            hidden  = drop(hidden)

            for block in blocks:
                attn_out, attn_w = block.attn(block.ln_1(hidden))[:2]
                hidden = hidden + attn_out
                hidden = hidden + block.mlp(block.ln_2(hidden))

                normed = ln_f(hidden)
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(normed)[0]
                else:
                    logits = normed[0] @ wte.weight.T

                probs   = torch.softmax(logits, dim=-1)
                top1_id = probs.argmax(dim=-1)
                top1_p  = probs.max(dim=-1).values

                layer_top1_ids.append(top1_id.numpy())
                layer_top1_probs.append(top1_p.numpy())

                # Ентропія logit на позиціях відповіді
                for pos in ans_positions:
                    if pos < probs.shape[0]:
                        ent = float(scipy_entropy(probs[pos].numpy() + 1e-12))
                        layer_entropies.append(ent)

                # Ентропія уваги
                avg_attn = attn_w[0].mean(dim=0).numpy()
                for pos in ans_positions:
                    if pos < avg_attn.shape[0]:
                        ent = float(scipy_entropy(avg_attn[pos] + 1e-12))
                        attn_entropies.append(ent)

        layer_top1_ids   = np.array(layer_top1_ids)    # (layers, full_len)
        layer_top1_probs = np.array(layer_top1_probs)

        # Метрики по токенах відповіді
        per_token_metrics = []
        for i, pos in enumerate(ans_positions):
            if i >= ans_len:
                break
            final_id     = full_ids[0, q_len + i].item()
            thought_ids  = layer_top1_ids[:, pos]
            thought_probs= layer_top1_probs[:, pos]
            agreements   = thought_ids == final_id

            per_token_metrics.append({
                "agreement_ratio":       float(agreements.mean()),
                "first_agreement_layer": int(np.argmax(agreements)) + 1
                                         if agreements.any() else num_layers,
                "flip_count":            int(np.sum(thought_ids[:-1] != thought_ids[1:])),
                "final_prob":            float(layer_top1_probs[-1, pos]),
                "last_layer_agreed":     bool(agreements[-1]),
            })

        if not per_token_metrics:
            return self._empty_metrics(num_layers)

        mean_agree   = float(np.mean([t["agreement_ratio"]       for t in per_token_metrics]))
        mean_layer   = float(np.mean([t["first_agreement_layer"] for t in per_token_metrics]))
        total_flips  = int(sum(t["flip_count"]                   for t in per_token_metrics))
        last_ratio   = float(np.mean([t["last_layer_agreed"]     for t in per_token_metrics]))
        mean_conf    = float(np.mean([t["final_prob"]            for t in per_token_metrics]))
        mean_logent  = float(np.mean(layer_entropies)) if layer_entropies else 0.0
        mean_attent  = float(np.mean(attn_entropies))  if attn_entropies  else 0.0

        # Вердикт
        if mean_agree >= 0.6:
            verdict = "CONSISTENT"
        elif last_ratio >= 0.6:
            verdict = "CONVERGED"
        elif total_flips > 2 * ans_len:
            verdict = "CHANGED_MIND"
        else:
            verdict = "UNCERTAIN"

        return {
            "agreement_ratio":       mean_agree,
            "mean_first_layer":      mean_layer,
            "total_flips":           total_flips,
            "last_layer_ratio":      last_ratio,
            "mean_final_confidence": mean_conf,
            "mean_logit_entropy":    mean_logent,
            "mean_attention_entropy":mean_attent,
            "verdict":               verdict,
        }

    def _empty_metrics(self, num_layers: int) -> dict:
        return {
            "agreement_ratio": 0.0, "mean_first_layer": float(num_layers),
            "total_flips": 0, "last_layer_ratio": 0.0,
            "mean_final_confidence": 0.0, "mean_logit_entropy": 0.0,
            "mean_attention_entropy": 0.0, "verdict": "UNKNOWN",
        }

    def run(self, dataset: list[dict],
            max_new_tokens: int = 15,
            verbose: bool = True) -> list[SampleResult]:
        """
        Запускає експеримент на датасеті.
        Повертає список SampleResult.
        """
        results = []
        n = len(dataset)

        print(f"\n  Запускаємо на {n} питаннях...")
        print(f"  {'#':>4}  {'Correct':>7}  {'Agree':>6}  "
              f"{'Flips':>6}  {'Conf':>5}  {'Verdict':<14}  Питання")
        print(f"  {'─'*80}")

        correct_count = 0

        for i, sample in enumerate(dataset):
            t0 = time.time()

            question = sample["question"]
            correct  = sample["answer"]
            aliases  = sample.get("aliases", [])
            category = sample.get("category", "general")

            # Генерація
            model_answer = self._generate_answer(question, max_new_tokens)

            # Перевірка правильності
            is_correct = int(check_answer(model_answer, correct, aliases))
            correct_count += is_correct

            # Метрики
            metrics = self._compute_metrics(question, model_answer)

            elapsed = time.time() - t0

            result = SampleResult(
                question=question,
                correct_answer=correct,
                model_answer=model_answer[:60],
                category=category,
                is_correct=is_correct,
                elapsed_sec=round(elapsed, 2),
                **{k: metrics[k] for k in metrics},
            )
            results.append(result)

            if verbose:
                icon    = "✓" if is_correct else "✗"
                verdict = metrics["verdict"][:12]
                q_short = question[:35]
                print(f"  {i+1:>4}  {icon:>7}  "
                      f"{metrics['agreement_ratio']:>6.2f}  "
                      f"{metrics['total_flips']:>6}  "
                      f"{metrics['mean_final_confidence']:>5.2f}  "
                      f"{verdict:<14}  {q_short}")

        accuracy = correct_count / n if n > 0 else 0
        print(f"\n  Accuracy: {correct_count}/{n} = {accuracy:.1%}")

        # Зберігаємо CSV
        csv_path = os.path.join(self.output_dir, "experiment_results.csv")
        self._save_csv(results, csv_path)
        print(f"  Збережено: {csv_path}")

        return results

    def _save_csv(self, results: list[SampleResult], path: str):
        if not results:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[fi.name for fi in fields(results[0])])
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
