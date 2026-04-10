"""
Dataset Loader — завантаження QA датасету для Задачі 3.

Використовуємо вбудований набір фактичних питань (не потребує інтернету).
Додатково підтримує завантаження з HuggingFace datasets якщо є з'єднання.

Кожен зразок:
    {
        "question": str,       # питання
        "answer":   str,       # правильна відповідь (еталон)
        "aliases":  list[str], # альтернативні правильні відповіді
        "category": str,       # категорія (geography, science, history...)
    }
"""

import re


# ── Вбудований датасет — 60 фактичних питань ──────────────────────
BUILTIN_QA = [
    # Geography
    {"question": "What is the capital of France?",
     "answer": "Paris", "aliases": ["paris"], "category": "geography"},
    {"question": "What is the capital of Germany?",
     "answer": "Berlin", "aliases": ["berlin"], "category": "geography"},
    {"question": "What is the capital of Japan?",
     "answer": "Tokyo", "aliases": ["tokyo"], "category": "geography"},
    {"question": "What is the capital of Italy?",
     "answer": "Rome", "aliases": ["rome", "roma"], "category": "geography"},
    {"question": "What is the capital of Spain?",
     "answer": "Madrid", "aliases": ["madrid"], "category": "geography"},
    {"question": "What is the capital of Australia?",
     "answer": "Canberra", "aliases": ["canberra"], "category": "geography"},
    {"question": "What is the capital of Brazil?",
     "answer": "Brasilia", "aliases": ["brasilia", "brasília"], "category": "geography"},
    {"question": "What is the capital of Canada?",
     "answer": "Ottawa", "aliases": ["ottawa"], "category": "geography"},
    {"question": "What is the largest ocean on Earth?",
     "answer": "Pacific", "aliases": ["pacific ocean", "the pacific"], "category": "geography"},
    {"question": "What is the longest river in the world?",
     "answer": "Nile", "aliases": ["the nile", "nile river"], "category": "geography"},

    # Science
    {"question": "What is the chemical symbol for water?",
     "answer": "H2O", "aliases": ["h2o"], "category": "science"},
    {"question": "What planet is closest to the Sun?",
     "answer": "Mercury", "aliases": ["mercury"], "category": "science"},
    {"question": "What is the speed of light in km/s?",
     "answer": "300000", "aliases": ["299792", "approximately 300000"], "category": "science"},
    {"question": "How many bones are in the human body?",
     "answer": "206", "aliases": ["206 bones"], "category": "science"},
    {"question": "What is the powerhouse of the cell?",
     "answer": "mitochondria", "aliases": ["the mitochondria", "mitochondrion"], "category": "science"},
    {"question": "What gas do plants absorb from the atmosphere?",
     "answer": "carbon dioxide", "aliases": ["co2", "CO2"], "category": "science"},
    {"question": "What is the atomic number of gold?",
     "answer": "79", "aliases": [], "category": "science"},
    {"question": "What is the hardest natural substance on Earth?",
     "answer": "diamond", "aliases": ["diamonds"], "category": "science"},
    {"question": "How many chromosomes do humans have?",
     "answer": "46", "aliases": ["46 chromosomes"], "category": "science"},
    {"question": "What is the most abundant gas in Earth's atmosphere?",
     "answer": "nitrogen", "aliases": ["nitrogen gas"], "category": "science"},

    # History
    {"question": "In what year did World War II end?",
     "answer": "1945", "aliases": [], "category": "history"},
    {"question": "In what year did the French Revolution begin?",
     "answer": "1789", "aliases": [], "category": "history"},
    {"question": "Who was the first President of the United States?",
     "answer": "George Washington", "aliases": ["washington"], "category": "history"},
    {"question": "In what year did Christopher Columbus reach America?",
     "answer": "1492", "aliases": [], "category": "history"},
    {"question": "Who wrote the Declaration of Independence?",
     "answer": "Thomas Jefferson", "aliases": ["jefferson"], "category": "history"},
    {"question": "In what year did the Berlin Wall fall?",
     "answer": "1989", "aliases": [], "category": "history"},
    {"question": "Who was the first man to walk on the Moon?",
     "answer": "Neil Armstrong", "aliases": ["armstrong"], "category": "history"},
    {"question": "In what year was the Titanic sunk?",
     "answer": "1912", "aliases": [], "category": "history"},
    {"question": "Who invented the telephone?",
     "answer": "Alexander Graham Bell",
     "aliases": ["graham bell", "bell"], "category": "history"},
    {"question": "Who painted the Mona Lisa?",
     "answer": "Leonardo da Vinci",
     "aliases": ["da vinci", "leonardo"], "category": "history"},

    # Math
    {"question": "What is 2 plus 2?",
     "answer": "4", "aliases": ["four"], "category": "math"},
    {"question": "What is the square root of 144?",
     "answer": "12", "aliases": ["twelve"], "category": "math"},
    {"question": "What is 7 multiplied by 8?",
     "answer": "56", "aliases": ["fifty six", "fifty-six"], "category": "math"},
    {"question": "What is 15 percent of 200?",
     "answer": "30", "aliases": ["thirty"], "category": "math"},
    {"question": "What is the value of pi to two decimal places?",
     "answer": "3.14", "aliases": ["3.14159"], "category": "math"},

    # Literature
    {"question": "Who wrote Romeo and Juliet?",
     "answer": "Shakespeare",
     "aliases": ["william shakespeare"], "category": "literature"},
    {"question": "Who wrote Harry Potter?",
     "answer": "J.K. Rowling",
     "aliases": ["rowling", "jk rowling", "joanne rowling"], "category": "literature"},
    {"question": "Who wrote 1984?",
     "answer": "George Orwell",
     "aliases": ["orwell"], "category": "literature"},
    {"question": "Who wrote The Great Gatsby?",
     "answer": "F. Scott Fitzgerald",
     "aliases": ["fitzgerald", "scott fitzgerald"], "category": "literature"},
    {"question": "Who wrote Hamlet?",
     "answer": "Shakespeare",
     "aliases": ["william shakespeare"], "category": "literature"},

    # Technology
    {"question": "Who founded Microsoft?",
     "answer": "Bill Gates",
     "aliases": ["gates", "bill gates and paul allen"], "category": "technology"},
    {"question": "Who founded Apple?",
     "answer": "Steve Jobs",
     "aliases": ["jobs", "steve jobs and steve wozniak"], "category": "technology"},
    {"question": "In what year was the World Wide Web invented?",
     "answer": "1989", "aliases": ["1991"], "category": "technology"},
    {"question": "Who invented the World Wide Web?",
     "answer": "Tim Berners-Lee",
     "aliases": ["berners-lee", "berners lee"], "category": "technology"},
    {"question": "What does CPU stand for?",
     "answer": "Central Processing Unit",
     "aliases": ["central processing unit"], "category": "technology"},

    # General knowledge
    {"question": "How many days are in a leap year?",
     "answer": "366", "aliases": ["366 days"], "category": "general"},
    {"question": "How many sides does a hexagon have?",
     "answer": "6", "aliases": ["six"], "category": "general"},
    {"question": "What is the largest planet in our solar system?",
     "answer": "Jupiter", "aliases": ["jupiter"], "category": "general"},
    {"question": "What language is spoken in Brazil?",
     "answer": "Portuguese", "aliases": ["portuguese"], "category": "general"},
    {"question": "How many continents are there?",
     "answer": "7", "aliases": ["seven"], "category": "general"},
    {"question": "What is the smallest country in the world?",
     "answer": "Vatican City",
     "aliases": ["vatican", "the vatican"], "category": "general"},
    {"question": "In what year was the first iPhone released?",
     "answer": "2007", "aliases": [], "category": "general"},
    {"question": "Who invented the light bulb?",
     "answer": "Thomas Edison",
     "aliases": ["edison"], "category": "general"},
    {"question": "What is the currency of Japan?",
     "answer": "Yen", "aliases": ["the yen", "japanese yen"], "category": "general"},
    {"question": "How many strings does a standard guitar have?",
     "answer": "6", "aliases": ["six"], "category": "general"},
    {"question": "What is the tallest mountain in the world?",
     "answer": "Mount Everest",
     "aliases": ["everest", "mt everest"], "category": "general"},
    {"question": "What animal is the symbol of the WWF?",
     "answer": "Giant Panda",
     "aliases": ["panda", "giant panda"], "category": "general"},
    {"question": "What is the chemical symbol for gold?",
     "answer": "Au", "aliases": ["au"], "category": "general"},
    {"question": "How many players are on a soccer team?",
     "answer": "11", "aliases": ["eleven"], "category": "general"},
]


def check_answer(model_answer: str, correct_answer: str,
                 aliases: list) -> bool:
    """
    Перевіряє чи відповідь моделі містить правильну відповідь.
    Нечутливо до регістру, перевіряє входження.
    """
    model_lower   = model_answer.lower().strip()
    correct_lower = correct_answer.lower().strip()

    # Пряме входження
    if correct_lower in model_lower:
        return True

    # Перевіряємо псевдоніми
    for alias in aliases:
        if alias.lower() in model_lower:
            return True

    # Числа — точне співпадіння після видалення пробілів
    numbers = re.findall(r'\d+\.?\d*', model_lower)
    if numbers and correct_lower in numbers:
        return True

    return False


class DatasetLoader:
    def __init__(self, source: str = "builtin", n_samples: int = None):
        """
        source: "builtin" — вбудований датасет
                "huggingface" — спробує завантажити з HF
        n_samples: обмежити кількість зразків
        """
        self.source    = source
        self.n_samples = n_samples

    def load(self) -> list[dict]:
        if self.source == "huggingface":
            return self._load_huggingface()
        return self._load_builtin()

    def _load_builtin(self) -> list[dict]:
        data = BUILTIN_QA.copy()
        if self.n_samples:
            data = data[:self.n_samples]
        print(f"  Завантажено {len(data)} питань (вбудований датасет)")
        return data

    def _load_huggingface(self) -> list[dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset("trivia_qa", "rc.nocontext",
                              split="validation[:200]", trust_remote_code=True)
            data = []
            for item in ds:
                aliases = item["answer"]["aliases"] or []
                data.append({
                    "question": item["question"],
                    "answer":   item["answer"]["value"],
                    "aliases":  aliases,
                    "category": "trivia",
                })
            if self.n_samples:
                data = data[:self.n_samples]
            print(f"  Завантажено {len(data)} питань (TriviaQA)")
            return data
        except Exception as e:
            print(f"  HuggingFace недоступний ({e}), використовую вбудований датасет")
            return self._load_builtin()
