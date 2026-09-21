"""
Script de Preparação e Geração de Dataset Imobiliário - São Paulo
Projeto Integrador 6 (SI-PI6-2026-T012)

Este módulo é responsável por:
1. Gerar/estruturar o dataset limpo de imóveis de São Paulo com série histórica (2013-2026).
2. Fornecer pipeline de ingestão para datasets brutos (ex: ITBI Prefeitura de SP / CSVs externos).
"""

import os
import json
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BAIRROS_FILE = os.path.join(DATA_DIR, "bairros_sp.json")
OUTPUT_DATASET = os.path.join(DATA_DIR, "dataset_sp_imoveis.csv")


def load_bairros():
    with open(BAIRROS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sp_sample_dataset(n_samples: int = 6500, random_state: int = 42) -> pd.DataFrame:
    """
    Gera uma base de dados realista de transações imobiliárias de São Paulo (2013 - 2026),
    respeitando perfis estatísticos de precificação hedônica:
    - Efeito de bairro/localização (m² base)
    - Depreciação ou valorização por ano de construção (idade do imóvel)
    - Impacto de vagas de garagem, dormitórios, banheiros e tipologia
    """
    np.random.seed(random_state)
    bairros = load_bairros()
    bairros_dict = {b["nome"]: b for b in bairros}
    nomes_bairros = list(bairros_dict.keys())

    tipos = ["Apartamento", "Casa", "Cobertura"]
    tipo_prob = [0.75, 0.20, 0.05]
    tipo_mult = {"Apartamento": 1.0, "Casa": 0.92, "Cobertura": 1.45}

    records = []
    
    for i in range(n_samples):
        bairro_nome = np.random.choice(nomes_bairros)
        bairro_info = bairros_dict[bairro_nome]
        tipo = np.random.choice(tipos, p=tipo_prob)

        # Ano de construção (1975 até 2025)
        # Mais concentração em anos recentes
        ano_construcao = int(np.random.choice(
            np.arange(1975, 2026),
            p=np.linspace(0.005, 0.035, 51) / np.sum(np.linspace(0.005, 0.035, 51))
        ))
        idade_imovel = 2026 - ano_construcao

        # Ano de transação/anúncio (2013 a 2026)
        ano_transacao = int(np.random.randint(2013, 2027))

        # Metragem quadrada com distribuição log-normal típica imobiliária
        if tipo == "Apartamento":
            area_m2 = int(np.clip(np.random.exponential(scale=45) + 32, 28, 380))
        elif tipo == "Casa":
            area_m2 = int(np.clip(np.random.exponential(scale=70) + 80, 60, 600))
        else: # Cobertura
            area_m2 = int(np.clip(np.random.exponential(scale=80) + 120, 110, 750))

        # Configuração de cômodos proporcional à área
        if area_m2 < 50:
            quartos = int(np.random.choice([1, 2], p=[0.75, 0.25]))
            banheiros = 1
            vagas = int(np.random.choice([0, 1], p=[0.6, 0.4]))
        elif area_m2 < 85:
            quartos = int(np.random.choice([2, 3], p=[0.70, 0.30]))
            banheiros = int(np.random.choice([1, 2], p=[0.4, 0.6]))
            vagas = int(np.random.choice([1, 2], p=[0.75, 0.25]))
        elif area_m2 < 140:
            quartos = int(np.random.choice([2, 3, 4], p=[0.15, 0.70, 0.15]))
            banheiros = int(np.random.choice([2, 3], p=[0.4, 0.6]))
            vagas = int(np.random.choice([1, 2, 3], p=[0.2, 0.65, 0.15]))
        else:
            quartos = int(np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2]))
            banheiros = int(np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2]))
            vagas = int(np.random.choice([2, 3, 4, 5], p=[0.2, 0.5, 0.2, 0.1]))

        # Fator de Idade: imóveis muito novos têm prêmio (lançamento), muito velhos depreciam suavemente
        if idade_imovel <= 3:
            fator_idade = 1.15
        elif idade_imovel <= 10:
            fator_idade = 1.05
        elif idade_imovel <= 25:
            fator_idade = 0.95
        elif idade_imovel <= 40:
            fator_idade = 0.88
        else:
            fator_idade = 0.82

        # Fator de Vagas (vaga extra em SP agrega muito valor)
        fator_vagas = 1.0 + (vagas * 0.05)

        # Fator do tipo
        fator_tipo = tipo_mult[tipo]

        # Variação histórica (de 2013 a 2026 houve valorização acumulada)
        fator_ano_transacao = 1.0 + ((ano_transacao - 2013) * 0.042)

        # Preço por metro quadrado ajustado
        preco_m2_estimado = (
            bairro_info["preco_m2_base"]
            * (fator_ano_transacao / (1 + (2026 - 2013) * 0.042)) # Normalizado para o presente
            * fator_idade
            * fator_tipo
            * fator_vagas
        )

        # Ruído de mercado (± 8%)
        ruido = np.random.normal(1.0, 0.06)
        preco_m2_final = round(preco_m2_estimado * ruido, 2)
        preco_total = round(preco_m2_final * area_m2, 2)

        records.append({
            "id": f"SP-{i+1:06d}",
            "estado": "SP",
            "cidade": "São Paulo",
            "bairro": bairro_nome,
            "zona": bairro_info["zona"],
            "tipo": tipo,
            "area_m2": area_m2,
            "quartos": quartos,
            "banheiros": banheiros,
            "vagas": vagas,
            "ano_construcao": ano_construcao,
            "idade_imovel": idade_imovel,
            "ano_transacao": ano_transacao,
            "preco_m2": preco_m2_final,
            "preco": preco_total
        })

    df = pd.DataFrame(records)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)

    print("[1/2] Gerando base inicial parametrizada de São Paulo...")
    df = generate_sp_sample_dataset(n_samples=7500)
    
    print(f"[2/2] Salvando em: {OUTPUT_DATASET}")
    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    
    print(f"Dataset criado com sucesso! {len(df)} registros imobiliários.")
    print(f"Colunas: {list(df.columns)}")
    print(df.head(3))


if __name__ == "__main__":
    main()
