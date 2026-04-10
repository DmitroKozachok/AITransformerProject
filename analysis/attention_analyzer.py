"""
Отримання та аналіз карт уваги
"""

import torch
import numpy as np
from utils.token_utils import clean_tokens


class AttentionAnalyzer:
    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer

    def get_attentions(self, text: str):
        """
        Токенізує текст і повертає (clean_tokens, attentions).
        attentions — список тензорів (batch, heads, seq, seq) для кожного шару.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        raw_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tokens = clean_tokens(raw_tokens)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return tokens, outputs.attentions

    def get_avg_layer_attention(self, attentions, layer_idx: int) -> np.ndarray:
        """Повертає усереднену по головах матрицю уваги для шару layer_idx."""
        return attentions[layer_idx][0].mean(dim=0).numpy()

    def get_head_attention(self, attentions, layer_idx: int, head_idx: int) -> np.ndarray:
        """Повертає матрицю уваги конкретної голови."""
        return attentions[layer_idx][0][head_idx].numpy()

    def print_top_connections(self, tokens, attentions, top_k: int = 10):
        """Виводить топ зв'язків уваги для кожного шару."""
        for layer_idx in range(len(attentions)):
            avg_attention = self.get_avg_layer_attention(attentions, layer_idx)
            np.fill_diagonal(avg_attention, 0)

            flat_idx = np.argsort(avg_attention.flatten())[-top_k:][::-1]
            print(f"\nШар {layer_idx + 1}:")
            for idx in flat_idx:
                row, col = divmod(idx, len(tokens))
                score = avg_attention[row, col]
                print(f"  '{tokens[row]}' → '{tokens[col]}': {score:.4f}")
