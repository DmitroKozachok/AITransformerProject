"""
Feature Extractor — витягує n-вимірний вектор ознак з карт
для кожного токена відповіді GPT-2.

Ознаки (на токен):
  [0]  logit_confidence     — max softmax prob на цій позиції (Logit Lens, останній шар)
  [1]  logit_entropy        — ентропія розподілу logits (невпевненість у виборі)
  [2]  attention_entropy    — середня ентропія уваги по всіх головах і шарах
  [3]  attention_to_self    — скільки уваги токен приділяє сам собі
  [4]  attention_to_question— скільки уваги токен приділяє токенам питання
  [5]  rollout_score        — сумарний rollout вплив на цей токен
  [6]  gradient_saliency    — норма градієнта по embedding цього токена
  [7]  ffn_activation_norm  — середня норма FFN активацій по шарах
  [8]  hidden_state_norm    — норма hidden state останнього шару
  [9]  layer_confidence_std — std впевненості logit lens по шарах (стабільність передбачення)
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


class FeatureExtractor:
    """
    Для заданого тексту (питання+відповідь) витягує матрицю ознак:
    shape = (seq_len, NUM_FEATURES=10)
    """

    NUM_FEATURES = 10
    FEATURE_NAMES = [
        "logit_confidence",
        "logit_entropy",
        "attention_entropy",
        "attention_to_self",
        "attention_to_question",
        "rollout_score",
        "gradient_saliency",
        "ffn_activation_norm",
        "hidden_state_norm",
        "layer_confidence_std",
    ]

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer

    # ── Приватні хелпери ──────────────────────────────────────────────

    def _forward_collect(self, input_ids):
        """
        Один forward pass: збирає attentions, hidden states, FFN activations.
        Повертає dict з усім потрібним.
        """
        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)
        seq_len = input_ids.shape[1]

        ffn_acts    = []
        hidden_list = []
        attn_list   = []

        # Реєструємо hooks на FFN
        hooks = []
        for block in blocks:
            storage = {}
            def make_hook(s):
                def h(m, inp, out):
                    s['act'] = out.detach()
                return h
            hooks.append(block.mlp.act.register_forward_hook(make_hook(storage)))
            ffn_acts.append(storage)

        with torch.no_grad():
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            hidden  = wte(input_ids) + wpe(pos_ids)
            hidden  = drop(hidden)

            for block in blocks:
                attn_out, attn_weights = block.attn(block.ln_1(hidden))[:2]
                hidden = hidden + attn_out
                hidden = hidden + block.mlp(block.ln_2(hidden))
                hidden_list.append(hidden[0].numpy())
                attn_list.append(attn_weights[0].numpy())   # (heads, seq, seq)

        for h in hooks:
            h.remove()

        # Фінальні logits через lm_head або wte
        with torch.no_grad():
            normed = ln_f(torch.tensor(hidden_list[-1]).unsqueeze(0))
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(normed)[0].numpy()
            else:
                logits = (normed[0] @ wte.weight.detach().T).numpy()

        return {
            "hidden_states": hidden_list,   # list of (seq, emb), per layer
            "attentions":    attn_list,      # list of (heads, seq, seq), per layer
            "ffn_acts":      [s.get('act', None) for s in ffn_acts],
            "logits":        logits,         # (seq, vocab)
        }

    def _compute_rollout(self, attn_list):
        """Attention Rollout → (seq, seq)."""
        seq_len = attn_list[0].shape[-1]
        rollout = np.eye(seq_len)
        for attn in attn_list:
            avg = attn.mean(axis=0)
            res = 0.5 * avg + 0.5 * np.eye(seq_len)
            res = res / res.sum(axis=-1, keepdims=True)
            rollout = res @ rollout
        return rollout

    def _compute_gradient_saliency(self, input_ids):
        """Gradient saliency: норма ∂loss/∂embed для кожного токена."""
        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            return np.zeros(seq_len)

        embeds = wte(input_ids).detach().clone().requires_grad_(True)
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        hidden  = embeds + wpe(pos_ids)
        hidden  = drop(hidden)

        for block in blocks:
            hidden = block(hidden)[0]

        hidden = ln_f(hidden)
        if hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(hidden)
        else:
            logits = hidden @ wte.weight.T

        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1], input_ids[0, 1:]
        )
        loss.backward()

        return embeds.grad[0].norm(dim=-1).detach().numpy()

    # ── Публічний API ─────────────────────────────────────────────────

    def extract(self, full_text: str, question_len: int) -> np.ndarray:
        """
        Повертає матрицю ознак shape=(seq_len, NUM_FEATURES).
        question_len — кількість токенів у питанні (для ознаки attention_to_question).
        """
        inputs  = self.tokenizer(full_text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        data    = self._forward_collect(inputs["input_ids"])
        rollout = self._compute_rollout(data["attentions"])
        grads   = self._compute_gradient_saliency(inputs["input_ids"])

        logits  = data["logits"]                   # (seq, vocab)
        probs   = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        features = np.zeros((seq_len, self.NUM_FEATURES))

        # [0] logit_confidence — max prob
        features[:, 0] = probs.max(axis=-1)

        # [1] logit_entropy
        features[:, 1] = np.array([
            scipy_entropy(probs[i] + 1e-12) for i in range(seq_len)
        ])

        # [2] attention_entropy — середня ентропія уваги
        all_attn_entropy = []
        for attn in data["attentions"]:          # (heads, seq, seq)
            for head in attn:                    # (seq, seq)
                all_attn_entropy.append([
                    scipy_entropy(head[i] + 1e-12) for i in range(seq_len)
                ])
        features[:, 2] = np.mean(all_attn_entropy, axis=0)

        # [3] attention_to_self — діагональ усередненої уваги
        avg_attn = np.mean([a.mean(axis=0) for a in data["attentions"]], axis=0)
        features[:, 3] = np.diag(avg_attn)

        # [4] attention_to_question — увага до токенів питання
        if question_len > 0:
            features[:, 4] = avg_attn[:, :question_len].sum(axis=-1)

        # [5] rollout_score — сума стовпця rollout
        features[:, 5] = rollout.sum(axis=0)

        # [6] gradient_saliency
        features[:, 6] = grads

        # [7] ffn_activation_norm — середня норма FFN по шарах
        ffn_norms = []
        for act_tensor in data["ffn_acts"]:
            if act_tensor is not None:
                ffn_norms.append(act_tensor[0].norm(dim=-1).numpy())
        if ffn_norms:
            features[:, 7] = np.mean(ffn_norms, axis=0)

        # [8] hidden_state_norm — норма останнього hidden state
        features[:, 8] = np.linalg.norm(data["hidden_states"][-1], axis=-1)

        # [9] layer_confidence_std — std впевненості logit lens по шарах
        wte, wpe, drop, blocks, ln_f = _get_parts(self.model)
        layer_confidences = []
        with torch.no_grad():
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            hidden  = wte(inputs["input_ids"]) + wpe(pos_ids)
            hidden  = drop(hidden)
            for block in blocks:
                hidden = block(hidden)[0]
                normed = ln_f(hidden)
                if hasattr(self.model, 'lm_head'):
                    lgt = self.model.lm_head(normed)[0]
                else:
                    lgt = normed[0] @ wte.weight.T
                conf = torch.softmax(lgt, dim=-1).max(dim=-1).values.numpy()
                layer_confidences.append(conf)
        features[:, 9] = np.std(layer_confidences, axis=0)

        return features

    def extract_for_response(self, response_info: dict) -> dict:
        """
        Витягує ознаки для відповіді GPT-2.
        response_info — словник з response_generator.generate().
        Повертає dict з features, токенами і діагностикою.
        """
        full_text    = response_info["full_text"]
        q_len        = len(response_info["question_tokens"])
        ans_start    = response_info["answer_start_idx"]

        all_features = self.extract(full_text, question_len=q_len)
        ans_features = all_features[ans_start:]   # тільки токени відповіді
        ans_tokens   = response_info["answer_tokens"]

        # Скорочуємо до довжини відповіді
        min_len      = min(len(ans_tokens), ans_features.shape[0])
        ans_features = ans_features[:min_len]
        ans_tokens   = ans_tokens[:min_len]

        return {
            "tokens":        ans_tokens,
            "features":      ans_features,          # (ans_len, 10)
            "all_features":  all_features,          # (full_len, 10)
            "feature_names": self.FEATURE_NAMES,
            "question":      response_info["question"],
            "answer":        response_info["answer"],
        }