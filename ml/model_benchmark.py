"""
Pipeline de Treinamento com Hold-out 10% Cru, GridSearchCV (5-Fold),
Comparativo de 5 Modelos (incluindo XGBoost e Rede Neural MLP) e Projeção Anual (2013-2030)
Projeto Integrador VI (SI-PI6-2026-T012) - Sprint 3
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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


def _clean_param_names(params: dict) -> dict:
    """Remove prefixos de Pipeline/TransformedTargetRegressor para exibição limpa."""
    cleaned = {}
    for k, v in params.items():
        clean_k = k.replace("regressor__regressor__", "").replace("regressor__", "")
        cleaned[clean_k] = list(v) if isinstance(v, tuple) else v
    return cleaned


def compute_yearly_trends_and_future(df: pd.DataFrame, future_years=(2027, 2028, 2029, 2030)):
    """
    Calcula a série histórica anual (2013-2026) por estado (SP e SC)
    e ajusta tendência linear para estimar os próximos anos (2027-2030).
    """
    serie_historica = {}
    projecao_futura = {}

    for uf in ["SP", "SC"]:
        df_uf = df[df["estado"] == uf]
        yearly = (
            df_uf.groupby("ano_transacao")
            .agg(
                preco_m2_medio=("preco_m2", "mean"),
                preco_m2_mediana=("preco_m2", "median"),
                preco_total_medio=("preco", "mean"),
                amostras=("id", "count"),
            )
            .reset_index()
            .sort_values("ano_transacao")
        )

        hist_list = []
        for _, row in yearly.iterrows():
            hist_list.append({
                "ano": int(row["ano_transacao"]),
                "preco_m2_medio": round(float(row["preco_m2_medio"]), 2),
                "preco_m2_mediana": round(float(row["preco_m2_mediana"]), 2),
                "preco_total_medio": round(float(row["preco_total_medio"]), 2),
                "amostras": int(row["amostras"]),
            })
        serie_historica[uf] = hist_list

        # Tendência temporal linear sobre os valores anuais
        years_arr = yearly["ano_transacao"].values.reshape(-1, 1)
        m2_arr = yearly["preco_m2_medio"].values
        preco_arr = yearly["preco_total_medio"].values

        reg_m2 = LinearRegression().fit(years_arr, m2_arr)
        reg_preco = LinearRegression().fit(years_arr, preco_arr)

        # Desvio residual da tendência para banda de confiança
        m2_resid_std = float(np.std(m2_arr - reg_m2.predict(years_arr)))
        preco_resid_std = float(np.std(preco_arr - reg_preco.predict(years_arr)))

        future_list = []
        for idx, f_year in enumerate(future_years, start=1):
            X_fut = np.array([[f_year]])
            pred_m2 = float(reg_m2.predict(X_fut)[0])
            pred_preco = float(reg_preco.predict(X_fut)[0])
            # Banda cresce levemente com o horizonte futuro
            margem_m2 = m2_resid_std * (1.0 + 0.25 * idx)
            margem_preco = preco_resid_std * (1.0 + 0.25 * idx)

            future_list.append({
                "ano": int(f_year),
                "preco_m2_estimado": round(pred_m2, 2),
                "preco_m2_min": round(pred_m2 - margem_m2, 2),
                "preco_m2_max": round(pred_m2 + margem_m2, 2),
                "preco_total_estimado": round(pred_preco, 2),
                "preco_total_min": round(pred_preco - margem_preco, 2),
                "preco_total_max": round(pred_preco + margem_preco, 2),
            })
        projecao_futura[uf] = future_list

    return serie_historica, projecao_futura


def run_gridsearch_benchmark(cv_folds=5, test_size=0.10):
    print("=" * 75)
    print(" PIPELINE DE TREINAMENTO: HOLD-OUT 10% CRU + GRIDSEARCHCV (5-FOLD)")
    print(" Modelos: Ridge | Decision Tree | Random Forest | XGBoost | Rede Neural MLP")
    print("=" * 75)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset não encontrado em {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Base total carregada: {len(df):,} registros (SP: {(df['estado']=='SP').sum():,} | SC: {(df['estado']=='SC').sum():,})")

    features_cat = ["estado", "cidade", "bairro", "tipo"]
    features_num = ["area_m2", "idade_imovel", "quartos", "banheiros", "vagas", "ano_transacao"]
    X = df[features_cat + features_num]
    y = df["preco"]

    # Separação do conjunto de teste 10% "CRU" isolado antes de qualquer GridSearch
    X_dev, X_test_raw, y_dev, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=df["estado"]
    )
    print(f"-> Divisão Hold-out: {len(X_dev):,} amostras (90% Desenvolvimento/GridSearchCV) | {len(X_test_raw):,} amostras (10% Teste Cru Isolado)")

    models_config = [
        {
            "id": "ridge",
            "nome": "Regressão Linear (Ridge - Baseline)",
            "tipo": "Linear Regularizado",
            "descricao_didatica": "Baseline paramétrico linear com regularização L2. Varia o coeficiente alpha no GridSearchCV, mas continua limitado por não captar não-linearidades complexas.",
            "estimator": Ridge(random_state=42),
            "param_grid": {
                "regressor__alpha": [0.1, 1.0, 10.0, 50.0]
            }
        },
        {
            "id": "decision_tree",
            "nome": "Árvore de Decisão (Decision Tree)",
            "tipo": "Árvore Simples",
            "descricao_didatica": "Modelo baseado em regras condicionais. O GridSearchCV ajusta profundidade máxima (max_depth) e divisão mínima (min_samples_split) para conter overfitting.",
            "estimator": DecisionTreeRegressor(random_state=42),
            "param_grid": {
                "regressor__max_depth": [8, 12, 16],
                "regressor__min_samples_split": [4, 10]
            }
        },
        {
            "id": "random_forest",
            "nome": "Random Forest (Ensemble Bagging)",
            "tipo": "Ensemble (Bagging)",
            "descricao_didatica": "Combina dezenas de árvores em subamostras aleatórias (Bagging). O GridSearchCV calibra quantidade de árvores, profundidade e critério de divisão.",
            "estimator": RandomForestRegressor(random_state=42, n_jobs=1),
            "param_grid": {
                "regressor__n_estimators": [80, 120],
                "regressor__max_depth": [12, 16],
                "regressor__min_samples_split": [4, 8]
            }
        },
        {
            "id": "xgboost",
            "nome": "XGBoost (Extreme Gradient Boosting)",
            "tipo": "Ensemble (Boosting)",
            "descricao_didatica": "Árvores sequenciais que corrigem os erros residuais das árvores anteriores via gradiente descendente. Alta performance em dados tabulares.",
            "estimator": XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
            "param_grid": {
                "regressor__n_estimators": [100, 150],
                "regressor__max_depth": [5, 7],
                "regressor__learning_rate": [0.05, 0.1]
            }
        },
        {
            "id": "mlp_neural_net",
            "nome": "Rede Neural MLP (Multi-Layer Perceptron)",
            "tipo": "Rede Neural Artificial",
            "descricao_didatica": "Perceptron Multicamadas com escalonamento do alvo (StandardScaler). O GridSearchCV busca melhor arquitetura de camadas ocultas, regularização L2 e taxa de aprendizado.",
            "estimator": TransformedTargetRegressor(
                regressor=MLPRegressor(max_iter=250, early_stopping=True, random_state=42),
                transformer=StandardScaler()
            ),
            "param_grid": {
                "regressor__regressor__hidden_layer_sizes": [(64, 32), (128, 64)],
                "regressor__regressor__alpha": [0.001, 0.01],
                "regressor__regressor__learning_rate_init": [0.005, 0.01]
            }
        }
    ]

    benchmark_results = []
    trained_best_pipelines = {}

    for cfg in models_config:
        print(f"\n[GridSearchCV] Otimizando {cfg['nome']}...")
        t0 = time.time()

        pipeline = Pipeline(steps=[
            ("preprocessor", get_preprocessor(features_cat, features_num)),
            ("regressor", cfg["estimator"])
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["param_grid"],
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_dev, y_dev)
        elapsed_time = time.time() - t0

        best_model = grid.best_estimator_
        trained_best_pipelines[cfg["id"]] = best_model

        # Avaliação no conjunto de teste 10% CRU (inédito)
        preds_raw = best_model.predict(X_test_raw)
        r2_raw = float(r2_score(y_test_raw, preds_raw))
        mae_raw = float(mean_absolute_error(y_test_raw, preds_raw))
        mape_raw = float(np.mean(np.abs((y_test_raw - preds_raw) / y_test_raw)) * 100.0)
        rmse_raw = float(np.sqrt(mean_squared_error(y_test_raw, preds_raw)))
        r2_cv_mean = float(grid.best_score_)

        clean_best_params = _clean_param_names(grid.best_params_)
        n_combinacoes = len(grid.cv_results_["params"])

        print(f" -> Combinações testadas : {n_combinacoes} (x {cv_folds} folds = {n_combinacoes * cv_folds} treinos)")
        print(f" -> Melhores Parâmetros  : {clean_best_params}")
        print(f" -> R² CV (90% Treino)   : {r2_cv_mean * 100:.2f}%")
        print(f" -> R² Teste Cru (10%)   : {r2_raw * 100:.2f}%")
        print(f" -> MAE Teste Cru        : R$ {mae_raw:,.2f}")
        print(f" -> MAPE Teste Cru       : {mape_raw:.2f}%")
        print(f" -> RMSE Teste Cru       : R$ {rmse_raw:,.2f}")
        print(f" -> Tempo Total          : {elapsed_time:.2f}s")

        benchmark_results.append({
            "id": cfg["id"],
            "nome": cfg["nome"],
            "tipo": cfg["tipo"],
            "descricao_didatica": cfg["descricao_didatica"],
            "combinacoes_grid": n_combinacoes,
            "melhores_parametros": clean_best_params,
            "r2_cv_medio": round(r2_cv_mean, 4),
            "r2_cv_pct": round(r2_cv_mean * 100, 2),
            "r2_medio": round(r2_raw, 4),
            "r2_pct": round(r2_raw * 100, 2),
            "mae_medio": round(mae_raw, 2),
            "mape_medio_pct": round(mape_raw, 2),
            "rmse_medio": round(rmse_raw, 2),
            "tempo_treino_seg": round(elapsed_time, 2)
        })

    # Ordena pelo melhor desempenho no conjunto de teste 10% cru (inédito)
    benchmark_results.sort(key=lambda x: x["r2_medio"], reverse=True)
    best_model_info = benchmark_results[0]
    best_id = best_model_info["id"]

    print("\n" + "=" * 75)
    print(f" CAMPEÃO NO CONJUNTO DE TESTE CRU (10%): {best_model_info['nome']}")
    print(f" R² Teste Cru: {best_model_info['r2_pct']}% | MAE: R$ {best_model_info['mae_medio']:,.2f} | MAPE: {best_model_info['mape_medio_pct']}%")
    print("=" * 75)

    # Série histórica anual (2013-2026) e estimação futura (2027-2030)
    serie_historica, projecao_futura = compute_yearly_trends_and_future(df)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_amostras": len(df),
        "amostras_sp": int((df["estado"] == "SP").sum()),
        "amostras_sc": int((df["estado"] == "SC").sum()),
        "n_folds": cv_folds,
        "holdout_info": {
            "treino_cv_pct": int((1.0 - test_size) * 100),
            "teste_cru_pct": int(test_size * 100),
            "amostras_treino_cv": len(X_dev),
            "amostras_teste_cru": len(X_test_raw),
            "estratificado_por": "estado"
        },
        "modelo_campeao": best_id,
        "modelos": benchmark_results,
        "serie_anual_historica": serie_historica,
        "projecao_futura_anual": projecao_futura
    }

    with open(BENCHMARK_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Salva o modelo campeão em produção (já treinado nos 90% e avaliado nos 10% crus)
    best_pipeline = trained_best_pipelines[best_id]
    joblib.dump({
        "pipeline": best_pipeline,
        "metrics": {
            "r2": best_model_info["r2_medio"],
            "r2_cv": best_model_info["r2_cv_medio"],
            "mae": best_model_info["mae_medio"],
            "mape_pct": best_model_info["mape_medio_pct"],
            "rmse": best_model_info["rmse_medio"],
            "total_samples": len(df),
            "test_raw_samples": len(X_test_raw),
            "model_name": best_model_info["nome"],
            "best_params": best_model_info["melhores_parametros"]
        },
        "mean_residual_pct": max(0.06, min(best_model_info["mape_medio_pct"] / 100.0, 0.14)),
        "benchmark_summary": benchmark_results
    }, MODEL_OUTPUT)

    print(f"Relatório salvo em: {BENCHMARK_OUTPUT}")
    print(f"Modelo salvo em   : {MODEL_OUTPUT}")
    return output_data


if __name__ == "__main__":
    run_gridsearch_benchmark(cv_folds=5, test_size=0.10)
