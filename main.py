"""
GPT-2 Attention Map Analysis — головний файл запуску
26 графіків: увага / активації / градієнти / представлення /
             logit lens / ваги / probing
"""

import os
from config.settings import Config
from analysis.model_loader import ModelLoader
from analysis.attention_analyzer import AttentionAnalyzer
from visualization.heatmaps import HeatmapVisualizer
from visualization.entropy_viz import EntropyVisualizer
from visualization.token_viz import TokenVisualizer
from visualization.comparison_viz import ComparisonVisualizer
from visualization.similarity_viz import SimilarityVisualizer
from visualization.rollout_viz import RolloutVisualizer
from visualization.head_patterns_viz import HeadPatternsVisualizer
from visualization.word_influence_viz import WordInfluenceVisualizer
from visualization.activation_viz import ActivationVisualizer
from visualization.gradient_viz import GradientVisualizer
from visualization.representations_viz import RepresentationsVisualizer
from visualization.logit_lens_viz import LogitLensVisualizer
from visualization.weights_viz import WeightsVisualizer
from visualization.probing_viz import ProbingVisualizer
from utils.plot_utils import wait_all, set_output_dir
from utils.results_saver import ResultsSaver


def main():
    print("=" * 55)
    print("  GPT-2 Attention Map Explorer  —  26 графіків")
    print("=" * 55)

    # 1. Папка результатів у корені проекту
    project_root = os.path.dirname(os.path.abspath(__file__))
    saver = ResultsSaver(base_dir=os.path.join(project_root, "results"))
    set_output_dir(saver.plots_path)

    # 2. Завантаження моделі
    loader = ModelLoader(Config.MODEL_NAME)
    model, tokenizer = loader.load()

    # 3. Токенізація + увага
    print(f"\nАналіз тексту: '{Config.TEXT}'")
    analyzer = AttentionAnalyzer(model, tokenizer)
    tokens, attentions = analyzer.get_attentions(Config.TEXT)
    print(f"Токени: {tokens}")
    print(f"Шарів: {len(attentions)}")

    print("\n=== ТОП зв'язки уваги ===")
    analyzer.print_top_connections(tokens, attentions, top_k=Config.TOP_K_CONNECTIONS)

    saver.save_all(Config, tokens, attentions)

    # ── Ініціалізація візуалізаторів ──────────────────────────────────
    heatmap_viz   = HeatmapVisualizer(Config)
    entropy_viz   = EntropyVisualizer(Config)
    token_viz     = TokenVisualizer(Config)
    compare_viz   = ComparisonVisualizer(model, tokenizer, Config)
    sim_viz       = SimilarityVisualizer(Config)
    rollout_viz   = RolloutVisualizer(Config)
    head_patterns = HeadPatternsVisualizer(Config, n_clusters=4)
    influence_viz = WordInfluenceVisualizer(Config)
    act_viz       = ActivationVisualizer(Config)
    grad_viz      = GradientVisualizer(Config)
    repr_viz      = RepresentationsVisualizer(Config)
    logit_viz     = LogitLensVisualizer(Config)
    weights_viz   = WeightsVisualizer(Config)
    probing_viz   = ProbingVisualizer(Config)

    print("\nБудуємо графіки...")

    # ── УВАГА ─────────────────────────────────────────────────────────
    print("  [01] Карти уваги — всі шари...")
    heatmap_viz.plot_all_layers(tokens, attentions)

    print("  [02] Карти уваги — всі голови...")
    heatmap_viz.plot_all_heads(tokens, attentions, layer_idx=Config.INSPECT_LAYER)

    print("  [03] Ентропія голів...")
    entropy_viz.plot_entropy_heatmap(attentions)

    print("  [04] Еволюція токена...")
    token_viz.plot_token_evolution(tokens, attentions, token_idx=Config.TARGET_TOKEN_IDX)

    print("  [05] Порівняння речень...")
    compare_viz.plot_comparison(Config.COMPARISON_TEXTS, layer_idx=Config.INSPECT_LAYER)

    print("  [06] Схожість між шарами...")
    sim_viz.plot_layer_similarity(attentions)

    print("  [07] Attention Rollout heatmap...")
    rollout_viz.plot_rollout(tokens, attentions)

    print("  [08] Attention Rollout per token...")
    rollout_viz.plot_rollout_per_token(tokens, attentions)

    print("  [09] Кластеризація голів...")
    head_patterns.plot_head_clusters(attentions)

    print("  [10] Середні патерни кластерів...")
    head_patterns.plot_cluster_avg_patterns(tokens, attentions)

    print("  [11] Вплив слова...")
    influence_viz.plot_word_influence(tokens, attentions, token_idx=Config.INFLUENCE_TOKEN_IDX)

    # ── АКТИВАЦІЇ ─────────────────────────────────────────────────────
    print("  [12] Активації FFN — всі шари...")
    _, activations = act_viz.get_activations(model, tokenizer, Config.TEXT)
    act_viz.plot_all_layers(tokens, activations, max_neurons=Config.MAX_NEURONS)

    print("  [13] Топ нейрони FFN...")
    act_viz.plot_top_neurons(tokens, activations,
                             layer_idx=Config.INSPECT_LAYER,
                             top_n=Config.TOP_NEURONS)

    # ── ГРАДІЄНТИ ─────────────────────────────────────────────────────
    print("  [14] Градієнтна важливість (3 методи)...")
    grad_viz.plot_saliency_comparison(model, tokenizer, Config.TEXT)

    print("  [15] Порівняння градієнтів між реченнями...")
    grad_viz.plot_gradient_heatmap(model, tokenizer, Config.COMPARISON_TEXTS)

    # ── ПРЕДСТАВЛЕННЯ ─────────────────────────────────────────────────
    print("  [16] PCA токенів по шарах...")
    repr_viz.plot_pca_layers(model, tokenizer, Config.TEXT)

    print("  [17] Cosine similarity токенів по шарах...")
    repr_viz.plot_token_similarity(model, tokenizer, Config.TEXT)

    print("  [18] Residual stream (attention vs FFN)...")
    repr_viz.plot_residual_stream(model, tokenizer, Config.TEXT)

    # ── LOGIT LENS ────────────────────────────────────────────────────
    print("  [19] Logit Lens...")
    logit_viz.plot_logit_lens(model, tokenizer, Config.TEXT)

    print("  [20] Топ-k передбачень по шарах...")
    logit_viz.plot_topk_per_layer(model, tokenizer, Config.TEXT,
                                  position=Config.LOGIT_LENS_POSITION,
                                  top_k=Config.LOGIT_LENS_TOP_K)

    # ── ВАГИ ──────────────────────────────────────────────────────────
    print("  [21] Норми ваг по шарах...")
    weights_viz.plot_weight_norms(model)

    print("  [22] SVD матриць уваги...")
    weights_viz.plot_svd_weights(model, layer_idx=Config.INSPECT_LAYER,
                                 top_k=Config.SVD_TOP_K)

    print("  [23] Теплові карти матриць ваг...")
    weights_viz.plot_weight_heatmap(model, layer_idx=Config.INSPECT_LAYER)

    # ── PROBING ───────────────────────────────────────────────────────
    print("  [24] Probing: позиція токена...")
    probing_viz.plot_position_probing(model, tokenizer, Config.PROBING_TEXTS)

    print("  [25] Probing: межа слова...")
    probing_viz.plot_word_boundary_probing(model, tokenizer, Config.PROBING_TEXTS)

    print("  [26] Causal Tracing...")
    probing_viz.plot_causal_tracing(model, tokenizer,
                                    Config.CLEAN_TEXT, Config.CORRUPTED_TEXT)

    # ── ПО КОЖНОМУ ШАРУ ──────────────────────────────────────────────
    print("\nБудуємо графіки по кожному шару...")

    print("  Активації FFN по шарах...")
    act_viz.plot_per_layer(tokens, activations, max_neurons=Config.MAX_NEURONS)

    print("  Gradient Saliency по шарах...")
    grad_viz.plot_per_layer(model, tokenizer, Config.TEXT)

    print("  PCA по шарах...")
    repr_viz.plot_pca_per_layer(model, tokenizer, Config.TEXT)
\
    print("  Cosine similarity по шарах...")
    repr_viz.plot_similarity_per_layer(model, tokenizer, Config.TEXT)

    print("  Residual stream по шарах...")
    repr_viz.plot_residual_per_layer(model, tokenizer, Config.TEXT)

    print("  Logit Lens по шарах...")
    logit_viz.plot_per_layer(model, tokenizer, Config.TEXT,
                             top_k=Config.LOGIT_LENS_TOP_K)

    print("  SVD ваг по шарах...")
    weights_viz.plot_svd_per_layer(model, top_k=Config.SVD_TOP_K)

    print("  Теплові карти ваг по шарах...")
    weights_viz.plot_weight_heatmap_per_layer(model)

    print("  Probing по шарах...")
    probing_viz.plot_probing_per_layer(model, tokenizer, Config.PROBING_TEXTS)

    print(f"\n✓ Всі графіки побудовано — переключайся між вікнами!")
    wait_all()


if __name__ == "__main__":
    main()