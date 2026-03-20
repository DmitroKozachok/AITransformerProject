"""
Завантаження моделі та токенізатора GPT-2.
Використовуємо GPT2LMHeadModel — має і generate(), і output_attentions.
Блоки доступні через model.transformer.h
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel


class ModelLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model      = None
        self.tokenizer  = None

    def load(self):
        """Завантажує та повертає (model, tokenizer)."""
        print(f"Завантаження моделі '{self.model_name}'...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model     = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            output_attentions=True,
        )
        self.model.eval()
        print("Модель успішно завантажена.")
        return self.model, self.tokenizer