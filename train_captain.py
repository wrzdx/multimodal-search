"""
train_captain.py — Отдельное обучение The Captain (LambdaMART).

Запуските ОДИН раз перед демонстрацией:
    python train_captain.py

После обучения модель сохранится в captain_lgbm.txt
и будет автоматически загружаться при streamlit run app.py.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd
from ltr_captain import TheCaptain, generate_training_data
from pipeline import AllianceRetriever


def main():
    print("=" * 60)
    print("Обучение The Captain (LambdaMART)")
    print("=" * 60)

    # Load alliance (without Captain)
    alliance = AllianceRetriever()

    # Generate training data
    print("\nГенерация обучающих данных (20 запросов)...")
    train_df, groups = generate_training_data(
        alliance.df,
        alliance.bm25,
        alliance.splade,
        alliance.dense_searcher,
        n_queries=20,
        top_k=50,
    )

    if len(train_df) == 0:
        print("Ошибка: не удалось сгенерировать обучающие данные.")
        return

    # Train
    captain = TheCaptain()
    captain.train(train_df, groups)

    print(f"\nГотово! Модель сохранена в captain_lgbm.txt")
    print("Теперь streamlit run app.py будет использовать LambdaMART.")


if __name__ == "__main__":
    main()