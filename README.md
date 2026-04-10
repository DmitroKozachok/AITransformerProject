# GPT-2 Attention Map Explorer — 26 графіків

## Встановлення і запуск

```bash
pip install -r requirements.txt
python main.py
```

## Структура проекту

```
gpt2_attention/
├── main.py
├── requirements.txt
├── results/                     # авто-створюється
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── plots/               # 26 PNG
│       ├── attention_data.csv
│       └── metadata.json
├── config/
│   └── settings.py              # всі параметри
├── analysis/
│   ├── model_loader.py
│   └── attention_analyzer.py
├── utils/
│   ├── plot_utils.py
│   ├── token_utils.py
│   └── results_saver.py
└── visualization/
    ├── heatmaps.py              # 01, 02
    ├── entropy_viz.py           # 03
    ├── token_viz.py             # 04
    ├── comparison_viz.py        # 05
    ├── similarity_viz.py        # 06
    ├── rollout_viz.py           # 07, 08
    ├── head_patterns_viz.py     # 09, 10
    ├── word_influence_viz.py    # 11
    ├── activation_viz.py        # 12, 13
    ├── gradient_viz.py          # 14, 15
    ├── representations_viz.py   # 16, 17, 18
    ├── logit_lens_viz.py        # 19, 20
    ├── weights_viz.py           # 21, 22, 23
    └── probing_viz.py           # 24, 25, 26
```

## Всі 26 графіків

### Увага (01–11)
| # | Файл | Що показує |
|---|------|-----------|
| 01 | `01_all_layers_heatmap` | Усереднені карти уваги всіх 12 шарів |
| 02 | `02_layer_all_heads` | Всі 12 голів одного шару |
| 03 | `03_entropy` | Ентропія уваги: шари × голови |
| 04 | `04_token_evolution` | Еволюція уваги одного токена |
| 05 | `05_comparison` | Порівняння карт уваги між реченнями |
| 06 | `06_layer_similarity` | Косинусна схожість між шарами |
| 07 | `07_attention_rollout` | Накопичений вплив через всі шари |
| 08 | `08_rollout_per_token` | Сумарний вплив кожного токена |
| 09 | `09_head_clusters` | PCA + KMeans кластеризація 144 голів |
| 10 | `10_cluster_avg_patterns` | Середній патерн кожного кластера |
| 11 | `11_word_influence` | Вплив конкретного слова по шарах |

### Активації FFN (12–13)
| # | Що показує |
|---|-----------|
| 12 | Активації нейронів FFN — всі шари (токени × нейрони) |
| 13 | Топ-N найактивніших нейронів конкретного шару |

### Градієнти (14–15)
| # | Що показує |
|---|-----------|
| 14 | Порівняння: Gradient Saliency / Grad×Input / Integrated Gradients |
| 15 | Heatmap важливості токенів між реченнями |

### Представлення hidden states (16–18)
| # | Що показує |
|---|-----------|
| 16 | PCA токенів: як змінюються вектори від шару до шару |
| 17 | Cosine similarity між токенами на кожному шарі |
| 18 | Residual stream: внесок Attention і FFN окремо |

### Logit Lens (19–20)
| # | Що показує |
|---|-----------|
| 19 | Передбачення моделі після кожного шару (не чекаючи фіналу) |
| 20 | Топ-k токенів для конкретної позиції по шарах |

### Ваги (21–23)
| # | Що показує |
|---|-----------|
| 21 | Норми W_Q, W_K, W_V, W_O, FFN по шарах |
| 22 | SVD матриць уваги — сингулярні значення |
| 23 | Теплові карти матриць ваг (PCA 32×32) |

### Probing (24–26)
| # | Що показує |
|---|-----------|
| 24 | Чи закодована позиція токена в hidden state? |
| 25 | Чи закодована межа слова в hidden state? |
| 26 | Causal Tracing — де модель "помічає" різницю між реченнями |
