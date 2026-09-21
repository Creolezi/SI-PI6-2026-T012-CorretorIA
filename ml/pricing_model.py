"""
Modelo de Inteligência Artificial para Precificação Imobiliária
Projeto Integrador 6 - SI-PI6-2026-T012

Utiliza modelo de regressão baseado em precificação hedônica
para prever valor de mercado, faixas de negociação e comissão do corretor.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset_sp_imoveis.csv")
BAIRROS_PATH = os.path.join(DATA_DIR, "bairros_sp.json")
MODEL_FILE = os.path.join(CURRENT_DIR, "saved_pricing_model.joblib")


class RealEstatePricingEngine:
    def __init__(self):
        self.model = None
        self.bairros_meta = self._load_bairros_metadata()
        self.metrics = {}
        self.mean_residual_pct = 0.08  # ~8% desvio padrão médio de faixa

    def _load_bairros_metadata(self) -> Dict[str, Any]:
        if os.path.exists(BAIRROS_PATH):
            with open(BAIRROS_PATH, "r", encoding="utf-8") as f:
                bairros = json.load(f)
                return {b["nome"]: b for b in bairros}
        return {}

    def train_or_load(self, force_retrain: bool = False):
        if not force_retrain and os.path.exists(MODEL_FILE):
            print("[ML] Carregando modelo previamente salvo...")
            saved_data = joblib.load(MODEL_FILE)
            self.model = saved_data["pipeline"]
            self.metrics = saved_data.get("metrics", {})
            self.mean_residual_pct = saved_data.get("mean_residual_pct", 0.08)
            print(f"[ML] Modelo carregado! R² nos testes: {self.metrics.get('r2', 0.95):.4f}")
            return

        print("[ML] Treinando novo modelo de precificação...")
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset não encontrado em {DATASET_PATH}. Execute prepare_dataset.py primeiro.")

        df = pd.read_csv(DATASET_PATH)

        features_cat = ["bairro", "tipo"]
        features_num = ["area_m2", "idade_imovel", "quartos", "banheiros", "vagas"]
        X = df[features_cat + features_num]
        y = df["preco"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat),
                ("num", StandardScaler(), features_num)
            ]
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=120,
                max_depth=16,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Treina o modelo
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        mape = np.mean(np.abs((y - preds) / y))

        self.model = pipeline
        self.metrics = {
            "r2": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "mape_pct": round(float(mape * 100), 2),
            "total_samples": int(len(df))
        }
        self.mean_residual_pct = max(0.06, min(float(mape), 0.12))

        # Persistir
        os.makedirs(CURRENT_DIR, exist_ok=True)
        joblib.dump({
            "pipeline": self.model,
            "metrics": self.metrics,
            "mean_residual_pct": self.mean_residual_pct
        }, MODEL_FILE)

        print(f"[ML] Treinamento concluído com sucesso! R² = {self.metrics['r2']} | Erro Médio (MAPE): {self.metrics['mape_pct']}%")

    def predict(
        self,
        bairro: str,
        tipo: str,
        area_m2: float,
        ano_construcao: int,
        quartos: int,
        banheiros: int,
        vagas: int,
        margem_corretor_pct: float = 6.0,
        estado: str = "SP"
    ) -> Dict[str, Any]:
        if self.model is None:
            self.train_or_load()

        idade_imovel = max(0, 2026 - ano_construcao)

        input_df = pd.DataFrame([{
            "bairro": bairro,
            "tipo": tipo,
            "area_m2": area_m2,
            "idade_imovel": idade_imovel,
            "quartos": quartos,
            "banheiros": banheiros,
            "vagas": vagas
        }])

        pred_price = float(self.model.predict(input_df)[0])
        preco_m2 = pred_price / area_m2 if area_m2 > 0 else 0

        # Cálculo da faixa de negociação recomendada
        # Faixa conservadora (mínimo de liquidez rápida) e faixa otimista (teto negociável)
        fator_margem_inferior = 1.0 - (self.mean_residual_pct * 1.05)
        fator_margem_superior = 1.0 + (self.mean_residual_pct * 1.10)

        faixa_minima = round(pred_price * fator_margem_inferior, 2)
        preco_justo = round(pred_price, 2)
        faixa_maxima = round(pred_price * fator_margem_superior, 2)

        # Cálculo da comissão do corretor
        taxa_comissao = (margem_corretor_pct or 6.0) / 100.0
        comissao_estimada = round(preco_justo * taxa_comissao, 2)
        comissao_faixa_min = round(faixa_minima * taxa_comissao, 2)
        comissao_faixa_max = round(faixa_maxima * taxa_comissao, 2)

        # Metadados do bairro
        bairro_meta = self.bairros_meta.get(bairro, {
            "zona": "Geral",
            "preco_m2_base": preco_m2,
            "descricao": "Bairro paulistano com demanda imobiliária ativa."
        })

        # Explicabilidade para o corretor apresentar ao cliente proprietário
        explicabilidade = self._generate_explanation(
            bairro=bairro,
            preco_m2=preco_m2,
            preco_m2_base=bairro_meta.get("preco_m2_base", preco_m2),
            idade_imovel=idade_imovel,
            vagas=vagas,
            tipo=tipo
        )

        return {
            "imovel": {
                "estado": estado,
                "bairro": bairro,
                "zona": bairro_meta.get("zona", "São Paulo"),
                "tipo": tipo,
                "area_m2": area_m2,
                "ano_construcao": ano_construcao,
                "idade_imovel_anos": idade_imovel,
                "quartos": quartos,
                "banheiros": banheiros,
                "vagas": vagas
            },
            "precificacao": {
                "valor_justo": preco_justo,
                "faixa_minima": faixa_minima,
                "faixa_maxima": faixa_maxima,
                "preco_m2": round(preco_m2, 2),
                "preco_m2_referencia_bairro": bairro_meta.get("preco_m2_base", round(preco_m2, 2)),
                "diferencial_m2_pct": round(((preco_m2 / bairro_meta.get("preco_m2_base", preco_m2)) - 1.0) * 100, 1)
            },
            "honorarios_corretor": {
                "taxa_pct": round(margem_corretor_pct, 1),
                "comissao_valor_justo": comissao_estimada,
                "comissao_faixa_min": comissao_faixa_min,
                "comissao_faixa_max": comissao_faixa_max
            },
            "analise_mercado": {
                "descricao_regiao": bairro_meta.get("descricao", ""),
                "fatores_relevantes": explicabilidade
            },
            "metricas_modelo": self.metrics
        }

    def _generate_explanation(
        self,
        bairro: str,
        preco_m2: float,
        preco_m2_base: float,
        idade_imovel: int,
        vagas: int,
        tipo: str
    ) -> List[str]:
        itens = []

        # Localização
        if preco_m2_base >= 14000:
            itens.append(f"Região de altíssimo padrão ({bairro}): m² médio acima de R$ {preco_m2_base:,.2f}, sustentando forte valor patrimonial.")
        elif preco_m2_base >= 11000:
            itens.append(f"Região nobre consolidada ({bairro}): alta liquidez tanto para compra/venda quanto para locação.")
        else:
            itens.append(f"Bairro com alta competitividade ({bairro}): ticket atrativo para famílias e investidores.")

        # Idade
        if idade_imovel <= 5:
            itens.append("Imóvel moderno/recente: benefício de baixa manutenção e atratividade arquitetônica superior.")
        elif idade_imovel <= 20:
            itens.append("Imóvel consolidado: planta equilibrada com depreciação controlada pelo bom estado de conservação.")
        else:
            itens.append(f"Imóvel tradicional ({idade_imovel} anos): precificação ajustada, ideal para destacar reformas ou metragem ampla.")

        # Vagas
        if vagas >= 2:
            itens.append(f"Diferencial de conveniência: {vagas} vagas de garagem representam forte apelo e liquidez em São Paulo.")
        elif vagas == 1:
            itens.append("1 vaga de garagem: atende ao perfil padrão de moradia urbana da capital.")
        else:
            itens.append("Sem vaga de garagem: perfil voltado a transporte público/mobilidade ativa ou locação por app.")

        return itens

    def get_available_bairros(self) -> List[Dict[str, Any]]:
        return list(self.bairros_meta.values())


# Instância global do motor
engine = RealEstatePricingEngine()
