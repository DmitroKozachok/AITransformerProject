"""
Console Reporter — детальний вивід у консоль.

Виводить всю інформацію про генерацію:
- кожен токен на кожному шарі
- топ-K кандидати на кожному кроці
- увагу
- thought vs answer порівняння
"""

import numpy as np


RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"


def _prob_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    bar    = "█" * filled + "░" * (width - filled)
    if prob >= 0.6:
        color = GREEN
    elif prob >= 0.3:
        color = YELLOW
    else:
        color = RED
    return f"{color}{bar}{RESET} {prob:.3f}"


def _tok_colored(tok: str, prob: float) -> str:
    if prob >= 0.6:
        return f"{GREEN}{BOLD}{tok}{RESET}"
    elif prob >= 0.3:
        return f"{YELLOW}{tok}{RESET}"
    else:
        return f"{RED}{tok}{RESET}"


class ConsoleReporter:

    def print_generation_full(self, gen_data: dict):
        """
        Виводить повну інформацію про генерацію:
        для кожного кроку — кожен шар + топ-K + увага.
        """
        steps      = gen_data["steps"]
        num_layers = gen_data["num_layers"]
        top_k      = gen_data["top_k"]

        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"{BOLD}{CYAN}  ПОКРОКОВА ГЕНЕРАЦІЯ — {len(steps)} токенів{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")
        print(f"  Питання : {CYAN}{gen_data['question']}{RESET}")
        print(f"  Відповідь: {GREEN}{BOLD}{gen_data['answer']}{RESET}")

        for step in steps:
            print(f"\n{BOLD}{'─'*70}{RESET}")
            print(f"{BOLD}  КРОК {step['step']:02d} → обрано: "
                  f"{GREEN}'{step['chosen_tok']}'{RESET}"
                  f"  впевненість: {_prob_bar(step['chosen_prob'])}")

            # Контекст
            ctx = " ".join(step["context"][-8:])
            print(f"  {DIM}Контекст (останні 8): ...{ctx}{RESET}")

            # Топ-K кандидати
            print(f"\n  {BOLD}{BLUE}Топ-{top_k} кандидати:{RESET}")
            for rank, (tok, prob) in enumerate(
                zip(step["topk_toks"], step["topk_probs"])
            ):
                marker = f"{GREEN}▶{RESET}" if rank == 0 else " "
                print(f"    {marker} {rank+1}. {_tok_colored(tok, prob):<20} "
                      f"{_prob_bar(prob)}")

            # Кожен шар
            print(f"\n  {BOLD}{MAGENTA}Думка по шарах:{RESET}")
            prev_tok = ""
            for layer_idx, (tok, prob) in enumerate(
                zip(step["layer_thoughts"], step["layer_probs"])
            ):
                changed = "  " if tok == prev_tok else f"{YELLOW}↺ {RESET}"
                is_final = tok == step["chosen_tok"]
                mark = f"{GREEN}✓{RESET}" if is_final else " "
                print(f"    {mark} L{layer_idx+1:02d} {changed}"
                      f"{_tok_colored(tok, prob):<20} "
                      f"{_prob_bar(prob, width=15)}")
                prev_tok = tok

            # Увага (топ-5 токенів)
            attn = step["attention"]
            if attn is not None:
                last_attn   = attn[-1]
                ctx_toks    = step["context"]
                top_attn_idx = np.argsort(last_attn)[-5:][::-1]
                print(f"\n  {BOLD}{CYAN}Увага (топ-5 токенів):{RESET}")
                for idx in top_attn_idx:
                    if idx < len(ctx_toks):
                        a = last_attn[idx]
                        print(f"    '{ctx_toks[idx]}'  "
                              f"{_prob_bar(min(a * 3, 1.0), width=15)} {a:.4f}")

        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"  {GREEN}✓ Фінальна відповідь: {BOLD}{gen_data['answer']}{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")

    def print_inner_speech_full(self, inner_data: dict):
        """
        Виводить внутрішній монолог:
        для кожного токена відповіді — кожен шар з топ-3 кандидатами.
        """
        a_tokens   = inner_data["a_tokens"]
        ppl        = inner_data["per_token_per_layer"]
        ans_len    = inner_data["ans_len"]
        num_layers = inner_data["num_layers"]

        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"{BOLD}{MAGENTA}  ВНУТРІШНІЙ МОНОЛОГ — токени відповіді по шарах{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")
        print(f"  Питання : {CYAN}{inner_data['question']}{RESET}")
        print(f"  Відповідь: {GREEN}{BOLD}{inner_data['answer']}{RESET}")

        for tok_idx in range(ans_len):
            final_tok = a_tokens[tok_idx] if tok_idx < len(a_tokens) else "?"
            print(f"\n{BOLD}  {'─'*60}{RESET}")
            print(f"{BOLD}  ТОКЕН {tok_idx+1}: {GREEN}'{final_tok}'{RESET}")

            prev_top1 = ""
            for layer_idx in range(num_layers):
                candidates = ppl[tok_idx][layer_idx]
                if candidates is None:
                    continue

                top1_tok  = candidates[0][0]
                top1_prob = candidates[0][1]
                changed   = f"{YELLOW}↺ {RESET}" if top1_tok != prev_top1 and prev_top1 else "  "
                is_final  = top1_tok == final_tok
                mark      = f"{GREEN}✓{RESET}" if is_final else " "

                # Топ-3 в одному рядку
                alts = "  |  ".join(
                    f"{_tok_colored(t, p)} {p:.2f}"
                    for t, p in candidates[:3]
                )
                print(f"  {mark} L{layer_idx+1:02d} {changed}{alts}")
                prev_top1 = top1_tok

        print(f"\n{BOLD}{'═'*70}{RESET}")

    def print_thought_vs_answer(self, cmp: dict):
        """
        Виводить порівняння думки і відповіді з деталями по токенах.
        """
        verdict    = cmp["match"]
        agree      = cmp["agreement"]
        by_layer   = cmp.get("by_layer", {})
        num_layers = cmp.get("num_layers", 12)
        detail     = cmp.get("detail", {})

        verdict_colors = {
            "CONSISTENT":   GREEN,
            "CONVERGED":    CYAN,
            "UNCERTAIN":    YELLOW,
            "CHANGED_MIND": RED,
        }
        vc = verdict_colors.get(verdict, RESET)

        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"{BOLD}{vc}  THOUGHT vs ANSWER{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")
        print(f"  Питання : {CYAN}{cmp['question']}{RESET}")
        print(f"  Відповідь: {GREEN}{BOLD}{cmp['answer'][:80]}{RESET}")

        # Думка на кожному шарі
        print(f"\n  {BOLD}Думка по шарах (Logit Lens на позиціях відповіді):{RESET}")
        checkpoints = sorted(set([
            1, num_layers//4, num_layers//2,
            num_layers*3//4, num_layers
        ]))
        for layer_idx in checkpoints:
            thought = by_layer.get(layer_idx, "")
            print(f"    L{layer_idx:02d}  {thought}")

        # По токенах
        per_token = detail.get("per_token", [])
        if per_token:
            print(f"\n  {BOLD}По токенах відповіді:{RESET}")
            print(f"  {'Токен':<12} {'Agree':>7}  {'Flips':>6}  "
                  f"{'FirstL':>7}  {'Conf':>6}  {'Думав?'}")
            print(f"  {'─'*60}")
            for t in per_token:
                icon = f"{GREEN}✓{RESET}" if t["final_was_thought"] else f"{RED}✗{RESET}"
                agree_c = GREEN if t["agreement_ratio"] >= 0.6 else \
                          YELLOW if t["agreement_ratio"] >= 0.3 else RED
                print(f"  {icon} '{t['token']}'  {agree_c}"
                      f"{t['agreement_ratio']:>6.0%}{RESET}  "
                      f"{t['flip_count']:>6}  "
                      f"L{t['first_agreement_layer']:>5}  "
                      f"{t['final_prob']:>6.2f}  "
                      f"{t['thought_tokens'][-1] if t['thought_tokens'] else '?'}")

        icons = {"CONSISTENT":"✓✓","CONVERGED":"✓","UNCERTAIN":"?","CHANGED_MIND":"⚠"}
        praise = {
            "CONSISTENT":   "Відмінно! Думка і відповідь збіглись.",
            "CONVERGED":    "Добре! Прийшла до думки у фінальних шарах.",
            "UNCERTAIN":    "Невпевнено. Часткове збігання.",
            "CHANGED_MIND": "Підозріло. Модель постійно передумувала.",
        }
        print(f"\n  {vc}{BOLD}{icons.get(verdict,'·')}  "
              f"{praise.get(verdict, verdict)}{RESET}")
        print(f"  Agreement: {vc}{agree:.0%}{RESET}  |  Verdict: {vc}{verdict}{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")

    def print_mri_summary(self, mri_data: dict):
        """
        МРТ у консолі: для кожного кроку відповіді —
        думка на ключових шарах у вигляді "сканування".
        """
        thoughts   = mri_data["layer_thoughts"]   # (layers, ans_len)
        probs      = mri_data["layer_probs"]       # (layers, ans_len)
        final_toks = mri_data["final_tokens"]
        num_layers = mri_data["num_layers"]
        ans_len    = len(final_toks)

        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"{BOLD}{BLUE}  МРТ ДУМКИ — сканування шарів{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")
        print(f"  Питання : {CYAN}{mri_data['question']}{RESET}")
        print(f"  Відповідь: {GREEN}{BOLD}{mri_data['answer']}{RESET}")

        checkpoints = sorted(set([
            0, num_layers//4-1, num_layers//2-1,
            num_layers*3//4-1, num_layers-1
        ]))

        # Заголовок таблиці
        header = f"  {'Шар':<8}" + "".join(
            f" {(final_toks[i] if i < ans_len else '?'):^10}"
            for i in range(min(ans_len, 10))
        )
        print(f"\n{BOLD}{header}{RESET}")
        print(f"  {'─'*70}")

        for layer_idx in checkpoints:
            row = f"  {f'L{layer_idx+1:02d}':<8}"
            for tok_idx in range(min(ans_len, 10)):
                if layer_idx < len(thoughts) and tok_idx < len(thoughts[layer_idx]):
                    t = thoughts[layer_idx][tok_idx]
                    p = probs[layer_idx][tok_idx] if tok_idx < len(probs[layer_idx]) else 0
                    final = final_toks[tok_idx] if tok_idx < ans_len else ""
                    is_match = t == final
                    color = GREEN if is_match else (YELLOW if p > 0.3 else RED)
                    row += f" {color}{t[:9]:^10}{RESET}"
                else:
                    row += f" {'?':^10}"
            print(row)

        # Фінальна відповідь
        print(f"  {'─'*70}")
        final_row = f"  {f'{GREEN}ВІДПОВІДЬ':^8}"
        for tok in final_toks[:10]:
            final_row += f" {GREEN}{BOLD}{tok[:9]:^10}{RESET}"
        print(final_row)
        print(f"{BOLD}{'═'*70}{RESET}")
