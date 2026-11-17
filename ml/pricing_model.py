"""
Modelo de Inteligência Artificial para Precificação Imobiliária - 6 Estados (SP, SC, MG, RJ, DF, PR)
Projeto Integrador 6 - SI-PI6-2026-T012

Suporta precificação em 6 estados brasileiros e estimação de valor futuro (2026-2035)
utilizando o modelo campeão selecionado após GridSearchCV + Hold-out 10% Cru.
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
MODEL_FILE = os.path.join(CURRENT_DIR, "saved_pricing_model.joblib")
BENCHMARK_FILE = os.path.join(CURRENT_DIR, "benchmark_results.json")

STATE_FILES = {
    "SP": ("bairros_sp.json", "São Paulo"),
    "SC": ("bairros_sc.json", "Florianópolis"),
    "MG": ("bairros_mg.json", "Belo Horizonte"),
    "RJ": ("bairros_rj.json", "Rio de Janeiro"),
    "DF": ("bairros_df.json", "Brasília"),
    "PR": ("bairros_pr.json", "Curitiba"),
}


class RealEstatePricingEngine:
    def __init__(self):
        self.model = None
        self.bairros_por_estado: Dict[str, List[Dict[str, Any]]] = {}
        self.bairros_meta = self._load_all_states()
        self.metrics = {}
        self.mean_residual_pct = 0.08

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _load_all_states(self) -> Dict[str, Any]:
        meta = {}
        for uf, (filename, default_city) in STATE_FILES.items():
            items = self._load_json(os.path.join(DATA_DIR, filename))
            for b in items:
                b["estado"] = uf
                b["cidade"] = b.get("cidade", default_city)
                meta[f"{uf}_{b['nome']}"] = b
                if b["nome"] not in meta:
                    meta[b["nome"]] = b
            self.bairros_por_estado[uf] = items
        return meta

    def train_or_load(self, force_retrain: bool = False):
        if not force_retrain and os.path.exists(MODEL_FILE):
            print("[ML] Carregando modelo campeão do benchmark...")
            saved_data = joblib.load(MODEL_FILE)
            self.model = saved_data["pipeline"]
            self.metrics = saved_data.get("metrics", {})
            self.mean_residual_pct = saved_data.get("mean_residual_pct", 0.08)
            print(f"[ML] Modelo carregado! {self.metrics.get('model_name', 'Modelo IA')} - R²: {self.metrics.get('r2', 0):.4f}")
            return

        print("[ML] Executando pipeline GridSearchCV (5-Fold) com Hold-out 10% cru...")
        from ml.model_benchmark import run_gridsearch_benchmark
        run_gridsearch_benchmark(cv_folds=5, test_size=0.10)
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
        cidade: str = None,
        ano_transacao: int = 2026,
        ano_estimacao: int = 2026
    ) -> Dict[str, Any]:
        if self.model is None:
            self.train_or_load()

        estado = estado.upper()
        idade_imovel = max(0, 2026 - ano_construcao)
        ano_alvo = max(2026, min(2035, int(ano_estimacao or 2026)))

        # Resolução do bairro e cidade
        meta_key = f"{estado}_{bairro}"
        bairro_meta = self.bairros_meta.get(meta_key, self.bairros_meta.get(bairro, {}))

        if not cidade:
            default_city = STATE_FILES.get(estado, ("", "São Paulo"))[1]
            cidade = bairro_meta.get("cidade", default_city)

        input_df = pd.DataFrame([{
            "estado": estado,
            "cidade": cidade,
            "bairro": bairro,
            "tipo": tipo,
            "area_m2": area_m2,
            "idade_imovel": idade_imovel,
            "quartos": quartos,
            "banheiros": banheiros,
            "vagas": vagas,
            "ano_transacao": min(ano_transacao, 2026)
        }])

        preco_base_2026 = float(self.model.predict(input_df)[0])

        # Taxa de valorização anual regional para estimação futura (2026-2035)
        taxa_valorizacao = float(bairro_meta.get("valorizacao_anual", 0.055))
        anos_futuros = max(0, ano_alvo - 2026)
        fator_ano_escolhido = (1.0 + taxa_valorizacao) ** anos_futuros

        pred_price = preco_base_2026 * fator_ano_escolhido
        preco_m2 = pred_price / area_m2 if area_m2 > 0 else 0.0

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

        # Preço de referência do m² ajustado para o ano alvo
        preco_m2_base = float(bairro_meta.get("preco_m2_base", preco_m2)) * fator_ano_escolhido
        diferencial_m2 = ((preco_m2 / preco_m2_base) - 1.0) * 100.0 if preco_m2_base > 0 else 0.0

        # Projeções anuais completas (2027 a 2035) ancoradas no valor base 2026
        estimativa_futuro = []
        for idx, ano_fut in enumerate(range(2027, 2036), start=1):
            fator_composto = (1.0 + taxa_valorizacao) ** idx
            valor_fut = round(preco_base_2026 * fator_composto, 2)
            preco_m2_fut = round(valor_fut / area_m2, 2) if area_m2 > 0 else 0.0
            estimativa_futuro.append({
                "ano": ano_fut,
                "valor_estimado": valor_fut,
                "preco_m2_estimado": preco_m2_fut,
                "valorizacao_acumulada_pct": round((fator_composto - 1.0) * 100.0, 1)
            })

        explicabilidade = self._generate_explanation(
            estado=estado,
            cidade=cidade,
            bairro=bairro,
            preco_m2_base=float(bairro_meta.get("preco_m2_base", preco_m2)),
            idade_imovel=idade_imovel,
            vagas=vagas,
            ano_alvo=ano_alvo,
            taxa_valorizacao=taxa_valorizacao,
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
                "vagas": vagas,
                "ano_estimacao": ano_alvo
            },
            "precificacao": {
                "valor_justo": preco_justo,
                "faixa_minima": faixa_minima,
                "faixa_maxima": faixa_maxima,
                "preco_m2": round(preco_m2, 2),
                "preco_m2_referencia_bairro": round(preco_m2_base, 2),
                "diferencial_m2_pct": round(diferencial_m2, 1)
            },
            "estimativa_futuro": {
                "ano_selecionado": ano_alvo,
                "valor_base_2026": round(preco_base_2026, 2),
                "valor_no_ano_selecionado": preco_justo,
                "valorizacao_ate_ano_selecionado_pct": round((fator_ano_escolhido - 1.0) * 100.0, 1),
                "taxa_valorizacao_anual_pct": round(taxa_valorizacao * 100.0, 2),
                "projecoes": estimativa_futuro
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
        preco_m2_base: float,
        idade_imovel: int,
        vagas: int,
        ano_alvo: int = 2026,
        taxa_valorizacao: float = 0.055,
    ) -> List[str]:
        itens = []

        # Localização e Estado
        desc_estado = {
            "SP": f"Mercado paulistano ({bairro}): alta densidade de transações e liquidez consolidada (m² ref. R$ {preco_m2_base:,.2f}).",
            "SC": f"Litoral/Polo Catarinense — {cidade} ({bairro}): forte valorização turística e patrimonial (m² ref. R$ {preco_m2_base:,.2f}).",
            "MG": f"Mercado Mineiro — {cidade} ({bairro}): região tradicional e valorizada em MG (m² ref. R$ {preco_m2_base:,.2f}).",
            "RJ": f"Mercado Fluminense — {cidade} ({bairro}): praça de alta procura nacional e internacional (m² ref. R$ {preco_m2_base:,.2f}).",
            "DF": f"Mercado Brasiliense — {cidade} ({bairro}): demanda estável impulsionada por renda média elevada (m² ref. R$ {preco_m2_base:,.2f}).",
            "PR": f"Mercado Paranaense — {cidade} ({bairro}): destaque em qualidade urbana e valorização contínua (m² ref. R$ {preco_m2_base:,.2f}).",
        }
        itens.append(desc_estado.get(estado, f"{cidade} ({bairro}): m² de referência R$ {preco_m2_base:,.2f}."))

        # Horizonte futuro selecionado
        if ano_alvo > 2026:
            ganho_pct = ((1.0 + taxa_valorizacao) ** (ano_alvo - 2026) - 1.0) * 100.0
            itens.append(f"Projeção futura para {ano_alvo}: estimativa considera valorização regional composta de {taxa_valorizacao*100:.1f}% a.a. (+{ganho_pct:.1f}% sobre 2026).")

        # Idade
        if idade_imovel <= 3:
            itens.append("Imóvel moderno/lançamento: benefício de baixa depreciação e arquitetura contemporânea.")
        elif idade_imovel <= 15:
            itens.append("Imóvel consolidado: planta funcional com manutenção equilibrada.")
        else:
            itens.append(f"Imóvel tradicional ({idade_imovel} anos): precificação ajustada, com potencial de modernização.")

        # Vagas
        if vagas >= 2:
            itens.append(f"Diferencial competitivo: {vagas} vagas de garagem elevam consideravelmente a liquidez.")
        elif vagas == 1:
            itens.append("1 vaga de garagem: atende ao padrão urbano predominante.")
        else:
            itens.append("Sem vaga de garagem: atrativo para público focado em mobilidade urbana e locação ágil.")

        return itens

    def get_available_bairros(self, estado: str = "SP") -> List[Dict[str, Any]]:
        return self.bairros_por_estado.get(estado.upper(), self.bairros_por_estado.get("SP", []))

    def get_benchmark_results(self) -> Dict[str, Any]:
        if os.path.exists(BENCHMARK_FILE):
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


# Instância global
engine = RealEstatePricingEngine()
