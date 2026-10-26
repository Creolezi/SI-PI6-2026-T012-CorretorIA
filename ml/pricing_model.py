"""
Modelo de Inteligência Artificial para Precificação Imobiliária - Multi-Estado (SP e SC)
Projeto Integrador 6 - SI-PI6-2026-T012 - Sprint 2

Suporta precificação em São Paulo e Santa Catarina utilizando o modelo campeão
selecionado após validação cruzada K-Fold (Random Forest).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset_sp_imoveis.csv")
BAIRROS_SP_PATH = os.path.join(DATA_DIR, "bairros_sp.json")
BAIRROS_SC_PATH = os.path.join(DATA_DIR, "bairros_sc.json")
MODEL_FILE = os.path.join(CURRENT_DIR, "saved_pricing_model.joblib")
BENCHMARK_FILE = os.path.join(CURRENT_DIR, "benchmark_results.json")


class RealEstatePricingEngine:
    def __init__(self):
        self.model = None
        self.bairros_sp = self._load_json(BAIRROS_SP_PATH)
        self.bairros_sc = self._load_json(BAIRROS_SC_PATH)
        self.bairros_meta = self._index_bairros()
        self.metrics = {}
        self.mean_residual_pct = 0.12

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _index_bairros(self) -> Dict[str, Any]:
        meta = {}
        for b in self.bairros_sp:
            b["estado"] = "SP"
            b["cidade"] = b.get("cidade", "São Paulo")
            meta[f"SP_{b['nome']}"] = b
            meta[b['nome']] = b

        for b in self.bairros_sc:
            b["estado"] = "SC"
            meta[f"SC_{b['nome']}"] = b
            if b['nome'] not in meta:
                meta[b['nome']] = b
        return meta

    def train_or_load(self, force_retrain: bool = False):
        if not force_retrain and os.path.exists(MODEL_FILE):
            print("[ML] Carregando modelo campeão do benchmark...")
            saved_data = joblib.load(MODEL_FILE)
            self.model = saved_data["pipeline"]
            self.metrics = saved_data.get("metrics", {})
            self.mean_residual_pct = saved_data.get("mean_residual_pct", 0.12)
            print(f"[ML] Modelo carregado! {self.metrics.get('model_name', 'Random Forest')} - R²: {self.metrics.get('r2', 0):.4f}")
            return

        print("[ML] Executando benchmark completo para treino do modelo...")
        from ml.model_benchmark import run_kfold_benchmark
        run_kfold_benchmark(n_splits=5)
        self.train_or_load(force_retrain=False)

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
        estado: str = "SP",
        cidade: str = None
    ) -> Dict[str, Any]:
        if self.model is None:
            self.train_or_load()

        estado = estado.upper()
        idade_imovel = max(0, 2026 - ano_construcao)

        # Resolução do bairro e cidade
        meta_key = f"{estado}_{bairro}"
        bairro_meta = self.bairros_meta.get(meta_key, self.bairros_meta.get(bairro, {}))
        
        if not cidade:
            cidade = bairro_meta.get("cidade", "São Paulo" if estado == "SP" else "Florianópolis")

        input_df = pd.DataFrame([{
            "estado": estado,
            "cidade": cidade,
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

        # Faixas estatísticas de negociação
        margem_inf = 1.0 - (self.mean_residual_pct * 1.05)
        margem_sup = 1.0 + (self.mean_residual_pct * 1.10)

        faixa_minima = round(pred_price * margem_inf, 2)
        preco_justo = round(pred_price, 2)
        faixa_maxima = round(pred_price * margem_sup, 2)

        # Comissão do consultor
        taxa = (margem_corretor_pct or 6.0) / 100.0
        comissao_estimada = round(preco_justo * taxa, 2)
        comissao_faixa_min = round(faixa_minima * taxa, 2)
        comissao_faixa_max = round(faixa_maxima * taxa, 2)

        # Preço de referência do m²
        preco_m2_base = bairro_meta.get("preco_m2_base", preco_m2)
        diferencial_m2 = ((preco_m2 / preco_m2_base) - 1.0) * 100.0 if preco_m2_base > 0 else 0.0

        # Explicabilidade
        explicabilidade = self._generate_explanation(
            estado=estado,
            cidade=cidade,
            bairro=bairro,
            preco_m2=preco_m2,
            preco_m2_base=preco_m2_base,
            idade_imovel=idade_imovel,
            vagas=vagas,
            tipo=tipo
        )

        return {
            "imovel": {
                "estado": estado,
                "cidade": cidade,
                "bairro": bairro,
                "zona": bairro_meta.get("zona", "Região Metropolitana"),
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
                "preco_m2_referencia_bairro": round(preco_m2_base, 2),
                "diferencial_m2_pct": round(diferencial_m2, 1)
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
        estado: str,
        cidade: str,
        bairro: str,
        preco_m2: float,
        preco_m2_base: float,
        idade_imovel: int,
        vagas: int,
        tipo: str
    ) -> List[str]:
        itens = []

        # Localização e Estado
        if estado == "SC":
            if cidade == "Balneário Camboriú":
                itens.append(f"Região de Balneário Camboriú ({bairro}): maior valorização litorânea e m² médio do Brasil (base de R$ {preco_m2_base:,.2f}/m²).")
            elif cidade == "Florianópolis":
                itens.append(f"Capital catarinense ({bairro}): forte demanda residencial e turística, com alta procura em temporadas.")
            elif cidade == "Itapema":
                itens.append(f"Itapema ({bairro}): destaque nacional em crescimento imobiliário e valorização contínua.")
            else:
                itens.append(f"{cidade} ({bairro}): polo econômico catarinense com sólida estabilidade de preços.")
        else:
            if preco_m2_base >= 14000:
                itens.append(f"Região de altíssimo padrão em SP ({bairro}): m² médio acima de R$ {preco_m2_base:,.2f}, sustentando forte valor patrimonial.")
            else:
                itens.append(f"Bairro consolidado em SP ({bairro}): liquidez equilibrada para locação e compra.")

        # Idade
        if idade_imovel <= 3:
            itens.append("Imóvel moderno/lançamento: benefício de baixa depreciação e arquitetura contemporânea.")
        elif idade_imovel <= 15:
            itens.append("Imóvel consolidado: planta funcional com manutenção em dia.")
        else:
            itens.append(f"Imóvel tradicional ({idade_imovel} anos): precificação ajustada, com potencial de reforma para valorização.")

        # Vagas
        if vagas >= 2:
            itens.append(f"Diferencial competitivo: {vagas} vagas de garagem elevam consideravelmente a liquidez.")
        elif vagas == 1:
            itens.append("1 vaga de garagem: perfil padrão de moradia urbana.")
        else:
            itens.append("Sem vaga de garagem: atrativo para locação ágil ou público sem veículo próprio.")

        return itens

    def get_available_bairros(self, estado: str = "SP") -> List[Dict[str, Any]]:
        estado = estado.upper()
        if estado == "SC":
            return self.bairros_sc
        return self.bairros_sp

    def get_benchmark_results(self) -> Dict[str, Any]:
        if os.path.exists(BENCHMARK_FILE):
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


# Instância global
engine = RealEstatePricingEngine()
