"""
Централізовані налаштування проекту
"""


class Config:
    # --- Модель ---
    MODEL_NAME = "gpt2"

    # --- Текст для аналізу ---
    TEXT = "Discrete mathematics is essential for programming."

    # --- Тексти для порівняння і probing ---
    COMPARISON_TEXTS = [
        "Discrete mathematics is essential for programming.",
        "The cat sat on the mat.",
        "Python is a programming language.",
    ]
    PROBING_TEXTS = [
        "Discrete mathematics is essential for programming.",
        "The cat sat on the mat.",
        "Python is a programming language.",
        "Neural networks learn from data.",
        "The dog runs in the park.",
        "Algorithms solve complex problems efficiently.",
    ]

    # --- Causal Tracing ---
    CLEAN_TEXT     = "Discrete mathematics is essential for programming."
    CORRUPTED_TEXT = "Discrete biology is essential for programming."

    # --- Параметри аналізу уваги ---
    TOP_K_CONNECTIONS   = 10
    INSPECT_LAYER       = 5    # 0-based
    TARGET_TOKEN_IDX    = 0
    INFLUENCE_TOKEN_IDX = 3
    LOGIT_LENS_POSITION = -1   # позиція для topk (−1 = остання)
    LOGIT_LENS_TOP_K    = 10

    # --- Параметри активацій ---
    MAX_NEURONS = 64
    TOP_NEURONS = 20

    # --- Параметри ваг ---
    SVD_TOP_K = 20

    # --- Параметри графіків ---
    CMAP               = "magma"
    FIGURE_SIZE        = (20, 15)
    ENTROPY_CMAP       = "YlOrRd"
    SIMILARITY_CMAP    = "coolwarm"
    FONT_SIZE_TITLE    = 22
    FONT_SIZE_SUBTITLE = 14
    GRID_COLS          = 4

    # --- Детектор відповідей ---
    DETECTOR_QUESTIONS = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is 2 plus 2?",
    ]
    DETECTOR_MAX_TOKENS = 20