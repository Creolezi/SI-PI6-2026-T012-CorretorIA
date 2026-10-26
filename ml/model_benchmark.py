"""
Pipeline de Validação Cruzada (K-Fold 5) & Comparativo de 3 Modelos de IA
Projeto Integrador VI (SI-PI6-2026-T012) - Sprint 2

Compara de forma didática três abordagens para precificação imobiliária:
1. Regressão Linear / Ridge (Baseline clássico)
2. Árvore de Decisão Simples (Modelo de regras não-linear)
3. Random Forest (Ensemble de Bagging - Campeão)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_sp_imoveis.csv")
BENCHMARK_OUTPUT = os.path.join(CURRENT_DIR, "benchmark_results.json")
MODEL_OUTPUT = os.path.join(CURRENT_DIR, "saved_pricing_model.joblib")


def get_preprocessor(cat_features, num_features):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), num_features)
        ]
    )


def run_kfold_benchmark(n_splits=5):
    print("=" * 70)
    print(" INICIANDO BENCHMARK COM VALIDAÇÃO CRUZADA (5-FOLD CROSS VALIDATION)")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset não encontrado em {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Base carregada: {len(df):,} registros (SP: {(df['estado']=='SP').sum():,} | SC: {(df['estado']=='SC').sum():,})")

    features_cat = ["estado", "cidade", "bairro", "tipo"]
    features_num = ["area_m2", "idade_imovel", "quartos", "banheiros", "vagas"]
    X = df[features_cat + features_num]
    y = df["preco"]

    models_config = [
        {
            "id": "ridge",
            "nome": "Regressão Linear (Ridge - Baseline)",
            "tipo": "Linear Regularizado",
            "descricao_didatica": "Modelo estatístico clássico que assume relação direta e proporcional entre atributos e preço. Rápido, mas tem dificuldade para capturar efeitos não-lineares (ex: valor desproporcional de vagas em áreas nobres).",
            "estimator": Ridge(alpha=1.0)
        },
        {
            "id": "decision_tree",
            "nome": "Árvore de Decisão (Decision Tree)",
            "tipo": "Árvore Simples",
            "descricao_didatica": "Segmenta o mercado através de regras if/else sequenciais. Consegue identificar nichos não-lineares, mas uma única árvore sofre com alta variância e risco de overfitting.",
            "estimator": DecisionTreeRegressor(max_depth=14, min_samples_split=6, random_state=42)
        },
        {
            "id": "random_forest",
            "nome": "Random Forest (Ensemble)",
            "tipo": "Ensemble (Bagging)",
            "descricao_didatica": "Combina 100 árvores de decisão treinadas em subconjuntos aleatórios de dados. A média das árvores cancela os ruídos individuais, reduzindo a variância e alcançando a maior acurácia.",
            "estimator": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1)
        }
    ]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    benchmark_results = []
    trained_pipelines = {}

    for cfg in models_config:
        model_name = cfg["nome"]
        print(f"\n[Avaliando] {model_name}...")

        scores_r2 = []
        scores_mae = []
        scores_mape = []
        scores_rmse = []
        fit_times = []

        fold_idx = 1
        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline(steps=[
                ("preprocessor", get_preprocessor(features_cat, features_num)),
                ("regressor", cfg["estimator"])
            ])

            t0 = time.time()
            pipeline.fit(X_train, y_train)
            fit_time = time.time() - t0

            preds = pipeline.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            scores_r2.append(r2)
            scores_mae.append(mae)
            scores_mape.append(mape)
            scores_rmse.append(rmse)
            fit_times.append(fit_time)

            fold_idx += 1

        mean_r2 = float(np.mean(scores_r2))
        std_r2 = float(np.std(scores_r2))
        mean_mae = float(np.mean(scores_mae))
        std_mae = float(np.std(scores_mae))
        mean_mape = float(np.mean(scores_mape))
        std_mape = float(np.std(scores_mape))
        mean_rmse = float(np.mean(scores_rmse))
        avg_time = float(np.mean(fit_times))

        # Treina o modelo final na base inteira para produção
        final_pipeline = Pipeline(steps=[
            ("preprocessor", get_preprocessor(features_cat, features_num)),
            ("regressor", cfg["estimator"])
        ])
        final_pipeline.fit(X, y)
        trained_pipelines[cfg["id"]] = final_pipeline

        print(f" -> R² Médio : {mean_r2 * 100:.2f}% (± {std_r2 * 100:.2f}%)")
        print(f" -> MAE Médio: R$ {mean_mae:,.2f}")
        print(f" -> MAPE Médio: {mean_mape:.2f}%")
        print(f" -> Tempo Médio de Treino: {avg_time:.2f}s")

        benchmark_results.append({
            "id": cfg["id"],
            "nome": cfg["nome"],
            "tipo": cfg["tipo"],
            "descricao_didatica": cfg["descricao_didatica"],
            "r2_medio": round(mean_r2, 4),
            "r2_std": round(std_r2, 4),
            "r2_pct": round(mean_r2 * 100, 2),
            "mae_medio": round(mean_mae, 2),
            "mae_std": round(std_mae, 2),
            "mape_medio_pct": round(mean_mape, 2),
            "mape_std_pct": round(std_mape, 2),
            "rmse_medio": round(mean_rmse, 2),
            "tempo_treino_seg": round(avg_time, 3),
            "folds_detalhes": [
                {
                    "fold": i + 1,
                    "r2": round(scores_r2[i], 4),
                    "mae": round(scores_mae[i], 2),
                    "mape_pct": round(scores_mape[i], 2)
                }
                for i in range(n_splits)
            ]
        })

    # Ordena pelo melhor R2
    benchmark_results.sort(key=lambda x: x["r2_medio"], reverse=True)
    best_model_info = benchmark_results[0]
    best_id = best_model_info["id"]

    print("\n" + "=" * 70)
    print(f" MODELO CAMPEÃO SELECIONADO: {best_model_info['nome']} (R²: {best_model_info['r2_pct']}%)")
    print("=" * 70)

    # Persiste relatório do benchmark
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_amostras": len(df),
        "amostras_sp": int((df["estado"] == "SP").sum()),
        "amostras_sc": int((df["estado"] == "SC").sum()),
        "n_folds": n_splits,
        "modelo_campeao": best_id,
        "modelos": benchmark_results
    }

    with open(BENCHMARK_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Salva o modelo campeão para produção
    best_pipeline = trained_pipelines[best_id]
    joblib.dump({
        "pipeline": best_pipeline,
        "metrics": {
            "r2": best_model_info["r2_medio"],
            "mae": best_model_info["mae_medio"],
            "mape_pct": best_model_info["mape_medio_pct"],
            "total_samples": len(df),
            "model_name": best_model_info["nome"]
        },
        "mean_residual_pct": best_model_info["mape_medio_pct"] / 100.0,
        "benchmark_summary": benchmark_results
    }, MODEL_OUTPUT)

    print(f"Relatório de Benchmark salvo em: {BENCHMARK_OUTPUT}")
    print(f"Modelo de produção salvo em: {MODEL_OUTPUT}")
    return output_data


if __name__ == "__main__":
    run_kfold_benchmark(n_splits=5)
