"""
Задача 3 — Зв'язок між "впевненістю думки" і якістю відповіді.

Гіпотеза: модель яка "вирішила" відповідь на ранньому шарі
і не змінювала думку дає більш точні відповіді.

Запуск: python task3_main.py
"""

import os
from config.settings import Config
from analysis.model_loader import ModelLoader
from analysis.dataset_loader import DatasetLoader
from analysis.experiment_runner import ExperimentRunner
from analysis.results_analyzer import ResultsAnalyzer
from utils.plot_utils import wait_all, set_output_dir
from utils.results_saver import ResultsSaver


def main():
    print("\033[1m" + "=" * 65 + "\033[0m")
    print("\033[1m\033[96m  ЗАДАЧА 3 — Впевненість думки vs Якість відповіді\033[0m")
    print("\033[1m" + "=" * 65 + "\033[0m")

    project_root = os.path.dirname(os.path.abspath(__file__))
    saver = ResultsSaver(base_dir=os.path.join(project_root, "results"))
    set_output_dir(saver.plots_path)

    # Завантаження моделі
    loader = ModelLoader(Config.MODEL_NAME)
    model, tokenizer = loader.load()

    # Завантаження датасету
    print("\n  Завантаження датасету...")
    dataset_loader = DatasetLoader(
        source=Config.TASK3_DATASET_SOURCE,
        n_samples=Config.TASK3_N_SAMPLES,
    )
    dataset = dataset_loader.load()

    # Запуск експерименту
    runner = ExperimentRunner(
        model=model,
        tokenizer=tokenizer,
        output_dir=saver.plots_path,
    )
    results = runner.run(
        dataset=dataset,
        max_new_tokens=Config.TASK3_MAX_NEW_TOKENS,
        verbose=True,
    )

    # Аналіз результатів
    analyzer = ResultsAnalyzer(Config)
    analyzer.run_full_analysis(results)

    print(f"\n\033[1m\033[92m✓ Готово! Графіки збережено у {saver.plots_path}\033[0m")
    wait_all()


if __name__ == "__main__":
    main()
