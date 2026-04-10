"""
GPT-2 — Generation Explorer з повним виводом у консоль
"""

import os
from config.settings import Config
from analysis.model_loader import ModelLoader
from analysis.response_generator import ResponseGenerator
from analysis.thought_vs_answer import ThoughtVsAnswerAnalyzer
from analysis.console_reporter import ConsoleReporter
from visualization.thought_viz import ThoughtVisualizer
from visualization.mri_viz import MRIVisualizer, _collect_mri_data
from visualization.inner_speech_viz import InnerSpeechVisualizer, _collect_inner_speech
from visualization.generation_viz import GenerationVisualizer, _generate_with_details
from utils.plot_utils import wait_all, set_output_dir
from utils.results_saver import ResultsSaver


def main():
    print("\033[1m" + "=" * 70 + "\033[0m")
    print("\033[1m\033[96m  GPT-2 — Generation Explorer\033[0m")
    print("\033[1m" + "=" * 70 + "\033[0m")

    project_root = os.path.dirname(os.path.abspath(__file__))
    saver = ResultsSaver(base_dir=os.path.join(project_root, "results"))
    set_output_dir(saver.plots_path)

    loader = ModelLoader(Config.MODEL_NAME)
    model, tokenizer = loader.load()

    thought_an  = ThoughtVsAnswerAnalyzer(model, tokenizer)
    thought_viz = ThoughtVisualizer(Config)
    mri_viz     = MRIVisualizer(Config)
    inner_viz   = InnerSpeechVisualizer(Config)
    gen_viz     = GenerationVisualizer(Config)
    reporter    = ConsoleReporter()

    for question in Config.DETECTOR_QUESTIONS:
        print(f"\n\033[1m\033[96m{'═'*70}\033[0m")
        print(f"\033[1m  Питання: {question}\033[0m")
        print(f"\033[1m{'═'*70}\033[0m")

        # ── Генерація з деталями (один прохід — використовуємо скрізь) ──
        print("\n  Генеруємо...")
        gen_data = _generate_with_details(
            model, tokenizer, question,
            max_new_tokens=Config.DETECTOR_MAX_TOKENS,
            top_k_show=Config.INNER_SPEECH_TOP_K,
        )
        answer = gen_data["answer"]

        # ── КОНСОЛЬ: покрокова генерація ──
        reporter.print_generation_full(gen_data)

        # ── КОНСОЛЬ: внутрішній монолог ──
        inner_data = _collect_inner_speech(
            model, tokenizer, question, answer,
            top_k=Config.INNER_SPEECH_TOP_K,
        )
        reporter.print_inner_speech_full(inner_data)

        # ── КОНСОЛЬ: МРТ ──
        mri_data = _collect_mri_data(model, tokenizer, question, answer)
        reporter.print_mri_summary(mri_data)

        # ── КОНСОЛЬ: Thought vs Answer ──
        cmp = thought_an.compare(question, answer)
        reporter.print_thought_vs_answer(cmp)

        # ── ГРАФІКИ ──
        print("\n  Будуємо графіки...")

        gen_viz.plot_generation_stream(gen_data)
        gen_viz.plot_step_by_step(gen_data)
        gen_viz.plot_generation_heatmap(gen_data)

        inner_viz.plot_topk_candidates(inner_data)
        inner_viz.plot_decision_moment(inner_data)
        inner_viz.plot_inner_monologue(inner_data)
        inner_viz.plot_probability_flow(inner_data)

        mri_viz.plot_thought_formation(mri_data)
        mri_viz.plot_attention_flow(mri_data)
        mri_viz.plot_hidden_trajectory(mri_data)
        mri_viz.plot_full_mri(mri_data)

        thought_viz.plot_all(cmp["detail"])

    print(f"\n\033[1m\033[92m✓ Готово!\033[0m")
    wait_all()


if __name__ == "__main__":
    main()
