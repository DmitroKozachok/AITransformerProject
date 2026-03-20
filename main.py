"""
GPT-2 — МРТ Думки + Thought vs Answer
"""

import os
from config.settings import Config
from analysis.model_loader import ModelLoader
from analysis.response_generator import ResponseGenerator
from analysis.thought_vs_answer import ThoughtVsAnswerAnalyzer
from visualization.thought_viz import ThoughtVisualizer
from visualization.mri_viz import MRIVisualizer
from utils.plot_utils import wait_all, set_output_dir
from utils.results_saver import ResultsSaver


def main():
    print("=" * 60)
    print("  GPT-2 — МРТ Думки + Thought vs Answer")
    print("=" * 60)

    project_root = os.path.dirname(os.path.abspath(__file__))
    saver = ResultsSaver(base_dir=os.path.join(project_root, "results"))
    set_output_dir(saver.plots_path)

    loader = ModelLoader(Config.MODEL_NAME)
    model, tokenizer = loader.load()

    generator   = ResponseGenerator(model, tokenizer, Config)
    thought_an  = ThoughtVsAnswerAnalyzer(model, tokenizer)
    thought_viz = ThoughtVisualizer(Config)
    mri_viz     = MRIVisualizer(Config)

    for question in Config.DETECTOR_QUESTIONS:
        print(f"\n{'═'*60}")
        print(f"  Питання: {question}")

        # Генеруємо відповідь
        response = generator.generate(question, max_new_tokens=Config.DETECTOR_MAX_TOKENS)
        answer   = response['answer']
        print(f"  Відповідь: {answer[:80]}")

        # Thought vs Answer — консоль + 4 графіки
        cmp = thought_an.compare(question, answer)
        thought_an.print_comparison(cmp)
        thought_viz.plot_all(cmp["detail"])

        # МРТ — 4 графіки
        print(f"\n  Будуємо МРТ думки...")
        mri_viz.plot_all(model, tokenizer, question, answer)

    print(f"\n✓ Готово! Всі графіки збережено.")
    wait_all()


if __name__ == "__main__":
    main()