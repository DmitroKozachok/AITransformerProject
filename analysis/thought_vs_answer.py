"""
Thought vs Answer Analyzer.

Правильна логіка для GPT-2:
  GPT-2 — авторегресивна мовна модель. Вона не "думає відповідь"
  до генерації. Натомість на кожному кроці генерації вона передбачає
  наступний токен.

  "Думка" = що модель передбачала на кожній позиції відповіді
             через проміжні шари (Logit Lens).

  "Відповідь" = токен який вона реально обрала (фінальний шар).

  Якщо проміжні шари вже передбачали той самий токен що і фінальний —
  модель була "впевнена" з раннього шару = CONSISTENT.

  Якщо тільки останні шари дали правильний токен = CONVERGED.
  Якщо постійно стрибала = CHANGED_MIND.
"""

import torch
import numpy as np
from utils.token_utils import clean_tokens


def _get_parts(model):
    if hasattr(model, 'h'):
        return model.wte, model.wpe, model.drop, model.h, model.ln_f
    t = model.transformer
    return t.wte, t.wpe, t.drop, t.h, t.ln_f


class ThoughtVsAnswerAnalyzer:
    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.model.eval()  # ✅ гарантує стабільну поведінку dropout/layernorm

    def analyze(self, question: str, answer: str) -> dict:
        """
        Для кожного токена відповіді дивимось що передбачали
        проміжні шари на тій самій позиції (Logit Lens).

        thought_token[layer] = топ-1 передбачення шару layer
                               на позиції що передує цьому токену
        final_token          = реально згенерований токен
        """
        self.model.eval()  # ✅ додатковий запобіжник перед кожним викликом

        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)
        num_layers = len(blocks)

        full_text   = question + answer
        inputs_q    = self.tokenizer(question,  return_tensors="pt")
        inputs_full = self.tokenizer(full_text, return_tensors="pt")

        q_len    = inputs_q["input_ids"].shape[1]
        full_ids = inputs_full["input_ids"]
        full_len = full_ids.shape[1]

        q_tokens = clean_tokens(
            self.tokenizer.convert_ids_to_tokens(full_ids[0, :q_len].tolist())
        )
        a_tokens = clean_tokens(
            self.tokenizer.convert_ids_to_tokens(full_ids[0, q_len:].tolist())
        )

        # Logit Lens: зберігаємо топ-1 і prob для кожного шару
        layer_top1_ids   = []  # (num_layers, full_len)
        layer_top1_probs = []

        with torch.no_grad():
            pos_ids = torch.arange(full_len).unsqueeze(0)
            hidden  = wte(full_ids) + wpe(pos_ids)
            hidden  = drop(hidden)

            for block in blocks:
                # ✅ Запобіжник: гарантуємо що hidden є 3D [batch, seq, dim]
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

                # ✅ Проходимо через шар трансформера
                hidden = block(hidden)[0]

                # ✅ Гарантуємо що після block hidden залишається 3D
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

                normed = ln_f(hidden)

                # Надійне обчислення логітів
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(normed)
                else:
                    logits = normed @ wte.weight.T

                # ✅ Знімаємо вимір батчу ТІЛЬКИ якщо він є (тобто тензор 3D)
                # Це гарантує, що logits завжди буде форми [seq_len, vocab_size]
                if logits.dim() == 3:
                    logits = logits[0]
                elif logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                probs   = torch.softmax(logits, dim=-1)
                top1_id = probs.argmax(dim=-1)
                top1_p  = probs.max(dim=-1).values

                # ✅ Гарантуємо 1D перед numpy
                if top1_id.dim() == 0:
                    top1_id = top1_id.unsqueeze(0)
                if top1_p.dim() == 0:
                    top1_p = top1_p.unsqueeze(0)

                layer_top1_ids.append(top1_id.numpy())
                layer_top1_probs.append(top1_p.numpy())

        layer_top1_ids   = np.array(layer_top1_ids)    # (layers, full_len)
        layer_top1_probs = np.array(layer_top1_probs)

        # Аналіз по кожному токену відповіді
        ans_len   = full_len - q_len
        per_token = []

        for i in range(ans_len):
            # Позиція що передбачає токен i відповіді
            pos      = q_len + i - 1
            if pos < 0:
                continue

            final_id  = full_ids[0, q_len + i].item()
            final_tok = a_tokens[i] if i < len(a_tokens) else "?"

            thought_ids   = layer_top1_ids[:, pos]
            thought_probs = layer_top1_probs[:, pos]
            thought_toks  = clean_tokens(
                self.tokenizer.convert_ids_to_tokens(thought_ids.tolist())
            )

            final_prob  = float(layer_top1_probs[-1, pos])
            agreements  = thought_ids == final_id
            agree_ratio = float(agreements.mean())
            first_agree = int(np.argmax(agreements)) if agreements.any() else -1
            flips       = int(np.sum(thought_ids[:-1] != thought_ids[1:]))

            per_token.append({
                "token":                 final_tok,
                "final_token":           final_tok,
                "thought_tokens":        thought_toks,
                "thought_probs":         thought_probs.tolist(),
                "final_prob":            final_prob,
                "agreement_ratio":       agree_ratio,
                "first_agreement_layer": first_agree + 1,
                "flip_count":            flips,
                "final_was_thought":     bool(agreements[-1]),
            })

        # Вердикт
        if not per_token:
            overall = {"verdict": "UNKNOWN"}
        else:
            mean_agree  = float(np.mean([t["agreement_ratio"]      for t in per_token]))
            total_flips = int(sum(t["flip_count"]                   for t in per_token))
            last_agreed = sum(1 for t in per_token if t["final_was_thought"])
            last_ratio  = last_agreed / len(per_token)

            layers_list = [t["first_agreement_layer"] for t in per_token
                           if t["first_agreement_layer"] > 0]
            mean_layer  = float(np.mean(layers_list)) if layers_list else num_layers

            if mean_agree >= 0.6:
                verdict = "CONSISTENT"
            elif last_ratio >= 0.6:
                verdict = "CONVERGED"
            elif total_flips > 2 * len(per_token):
                verdict = "CHANGED_MIND"
            else:
                verdict = "UNCERTAIN"

            overall = {
                "mean_agreement":   round(mean_agree, 3),
                "mean_first_layer": round(mean_layer, 1),
                "total_flips":      total_flips,
                "last_layer_ratio": round(last_ratio, 3),
                "verdict":          verdict,
            }

        return {
            "question":        question,
            "answer":          answer,
            "question_tokens": q_tokens,
            "answer_tokens":   a_tokens[:len(per_token)],
            "num_layers":      num_layers,
            "per_token":       per_token,
            "overall":         overall,
        }

    def get_thought_by_layer(self, question: str, answer: str) -> dict:
        """
        Повертає "думку" моделі як текст для ключових шарів.

        Думка шару N = токени відповіді які шар N передбачав
                       (топ-1 logit lens на позиціях відповіді).

        Це відповідає на питання:
        "Якби модель зупинилась після шару N — яку б відповідь вона дала?"
        """
        self.model.eval()  # ✅ додатковий запобіжник перед кожним викликом

        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)
        num_layers = len(blocks)

        full_text   = question + answer
        inputs_q    = self.tokenizer(question,  return_tensors="pt")
        inputs_full = self.tokenizer(full_text, return_tensors="pt")

        q_len    = inputs_q["input_ids"].shape[1]
        full_ids = inputs_full["input_ids"]
        full_len = full_ids.shape[1]

        layer_thoughts = {}

        with torch.no_grad():
            pos_ids = torch.arange(full_len).unsqueeze(0)
            hidden  = wte(full_ids) + wpe(pos_ids)
            hidden  = drop(hidden)

            for layer_idx, block in enumerate(blocks):
                # 1. Запобіжник: гарантуємо, що hidden є 3D [batch, seq, dim]
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

                # 2. Проходимо через шар трансформера
                hidden = block(hidden)[0]

                # ✅ 3. Гарантуємо що після block hidden залишається 3D
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

                normed = ln_f(hidden)

                # 4. Надійне обчислення логітів
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(normed)
                else:
                    logits = normed @ wte.weight.T

                # 5. Запобіжник: логіти для argmax мають бути 2D [seq_len, vocab_size]
                if logits.dim() == 3:
                    logits = logits[0]
                elif logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                # 6. Знаходимо топ-1 токени (гарантуємо 1D тензор [seq_len])
                seq_argmax = logits.argmax(dim=-1)
                if seq_argmax.dim() == 0:
                    seq_argmax = seq_argmax.unsqueeze(0)

                # 7. Безпечно витягуємо потрібні позиції
                ans_positions = list(range(q_len - 1, full_len - 1))
                top1_ids = [
                    seq_argmax[p].item()
                    for p in ans_positions
                    if p < seq_argmax.size(0)
                ]

                thought = self.tokenizer.decode(top1_ids, skip_special_tokens=True).strip()
                layer_thoughts[layer_idx + 1] = thought

        return {
            "by_layer":      layer_thoughts,
            "final_thought": layer_thoughts[num_layers],
            "early_thought": layer_thoughts.get(num_layers // 4, ""),
            "mid_thought":   layer_thoughts.get(num_layers // 2, ""),
            "question":      question,
            "answer":        answer.strip(),
        }

    def compare(self, question: str, answer: str) -> dict:
        """Зручний метод: думка + аналіз разом."""
        thought_info = self.get_thought_by_layer(question, answer)
        result       = self.analyze(question, answer)

        return {
            "question":    question,
            "thought":     thought_info["final_thought"],
            "mid_thought": thought_info["mid_thought"],
            "answer":      answer.strip(),
            "match":       result["overall"].get("verdict", "UNKNOWN"),
            "agreement":   result["overall"].get("mean_agreement", 0),
            "by_layer":    thought_info["by_layer"],
            "num_layers":  result["num_layers"],
            "detail":      result,
        }

    def print_comparison(self, cmp: dict):
        """Виводить думку і відповідь поруч."""
        verdict    = cmp["match"]
        agree      = cmp["agreement"]
        num_layers = cmp.get("num_layers", 12)
        by_layer   = cmp.get("by_layer", {})

        icons = {
            "CONSISTENT":   "✓✓",
            "CONVERGED":    "✓",
            "UNCERTAIN":    "?",
            "CHANGED_MIND": "⚠",
        }
        praise = {
            "CONSISTENT":   "Відмінно! Думка і відповідь збіглись з раннього шару.",
            "CONVERGED":    "Добре! Думка сформувалась у фінальних шарах і збіглась.",
            "UNCERTAIN":    "Невпевнено. Часткове збігання — можливі неточності.",
            "CHANGED_MIND": "Підозріло. Модель постійно передумувала.",
        }

        early = by_layer.get(max(1, num_layers // 4), "")
        mid   = by_layer.get(num_layers // 2, "")
        final = by_layer.get(num_layers, cmp["thought"])
        ans   = cmp["answer"][:80]

        print(f"\n{'─'*58}")
        print(f"  Питання         : {cmp['question']}")
        print(f"  Думка (L{max(1,num_layers//4):02d}/ранн.) : {early}")
        print(f"  Думка (L{num_layers//2:02d}/серед.) : {mid}")
        print(f"  Думка (L{num_layers:02d}/фінал) : {final}")
        print(f"  Відповідь       : {ans}")
        print(f"\n  {icons.get(verdict,'·')}  {praise.get(verdict, verdict)}")
        print(f"  Agreement: {agree:.0%}  |  Verdict: {verdict}")
        print(f"{'─'*58}")

    def print_report(self, result: dict):
        print(f"\n{'═'*58}")
        print(f"  THOUGHT vs ANSWER REPORT")
        print(f"{'═'*58}")
        print(f"  Питання : {result['question']}")
        print(f"  Відповідь: {result['answer'][:80]}")
        print(f"  Вердикт : {result['overall'].get('verdict','?')}")
        print(f"  Метрики : {result['overall']}")
        print(f"\n  По токенах:")
        for t in result["per_token"]:
            icon = "✓" if t["final_was_thought"] else "↺"
            print(f"    {icon} '{t['token']}'"
                  f"  agree={t['agreement_ratio']:.0%}"
                  f"  flips={t['flip_count']}"
                  f"  first_L={t['first_agreement_layer']}"
                  f"  conf={t['final_prob']:.2f}")
        print(f"{'═'*58}")