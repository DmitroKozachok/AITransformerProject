"""
Генерація відповіді GPT-2 на питання.
"""

import torch
from utils.token_utils import clean_tokens


class ResponseGenerator:
    def __init__(self, model, tokenizer, config):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = config

    def generate(self, question: str, max_new_tokens: int = 30) -> dict:
        """
        Генерує відповідь на питання і повертає:
        {
          'question': str,
          'answer': str,
          'full_text': str,         # питання + відповідь
          'question_tokens': list,
          'answer_tokens': list,
          'answer_start_idx': int,  # індекс першого токена відповіді
        }
        """
        inputs     = self.tokenizer(question, return_tensors="pt")
        input_len  = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,       # greedy для детермінованості
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_ids    = output_ids[0]
        answer_ids  = full_ids[input_len:]

        full_text   = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        q_tokens = clean_tokens(
            self.tokenizer.convert_ids_to_tokens(full_ids[:input_len].tolist())
        )
        a_tokens = clean_tokens(
            self.tokenizer.convert_ids_to_tokens(answer_ids.tolist())
        )

        return {
            "question":         question,
            "answer":           answer_text,
            "full_text":        full_text,
            "question_tokens":  q_tokens,
            "answer_tokens":    a_tokens,
            "answer_start_idx": input_len,
        }
