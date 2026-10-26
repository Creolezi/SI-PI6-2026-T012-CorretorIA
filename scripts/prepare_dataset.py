"""
Script de Preparação e Geração de Dataset Imobiliário - São Paulo & Santa Catarina
Projeto Integrador 6 (SI-PI6-2026-T012) - Sprint 2

Gera e unifica bases de dados imobiliárias de São Paulo (SP) e Santa Catarina (SC)
cobrindo a série histórica de 2013 a 2026 com base em parâmetros reais de mercado (FipeZAP/ITBI).
"""

import os
import json
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BAIRROS_SP_FILE = os.path.join(DATA_DIR, "bairros_sp.json")
BAIRROS_SC_FILE = os.path.join(DATA_DIR, "bairros_sc.json")
OUTPUT_DATASET = os.path.join(DATA_DIR, "dataset_sp_imoveis.csv")


def load_bairros(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_state_data(state_uf, state_name, bairros_list, n_samples, start_id, random_seed):
    np.random.seed(random_seed)
    bairros_dict = {b["nome"]: b for b in bairros_list}
    nomes_bairros = list(bairros_dict.keys())

    tipos = ["Apartamento", "Casa", "Cobertura"]
    # Em SC (especialmente litoral), coberturas e apartamentos têm proporção maior
    if state_uf == "SC":
        tipo_prob = [0.72, 0.20, 0.08]
    else:
        tipo_prob = [0.75, 0.20, 0.05]

    tipo_mult = {"Apartamento": 1.0, "Casa": 0.92, "Cobertura": 1.45}
    records = []

    for i in range(n_samples):
        bairro_nome = np.random.choice(nomes_bairros)
        bairro_info = bairros_dict[bairro_nome]
        cidade_nome = bairro_info.get("cidade", state_name)
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

        # Metragem quadrada com distribuição típica
        if tipo == "Apartamento":
            area_m2 = int(np.clip(np.random.exponential(scale=48) + 35, 30, 420))
        elif tipo == "Casa":
            area_m2 = int(np.clip(np.random.exponential(scale=75) + 85, 60, 650))
        else:  # Cobertura
            area_m2 = int(np.clip(np.random.exponential(scale=85) + 130, 120, 800))

        # Cômodos proporcionais à metragem
        if area_m2 < 55:
            quartos = int(np.random.choice([1, 2], p=[0.70, 0.30]))
            banheiros = 1
            vagas = int(np.random.choice([0, 1], p=[0.55, 0.45]))
        elif area_m2 < 90:
            quartos = int(np.random.choice([2, 3], p=[0.65, 0.35]))
            banheiros = int(np.random.choice([1, 2], p=[0.35, 0.65]))
            vagas = int(np.random.choice([1, 2], p=[0.70, 0.30]))
        elif area_m2 < 150:
            quartos = int(np.random.choice([2, 3, 4], p=[0.10, 0.75, 0.15]))
            banheiros = int(np.random.choice([2, 3], p=[0.35, 0.65]))
            vagas = int(np.random.choice([1, 2, 3], p=[0.15, 0.70, 0.15]))
        else:
            quartos = int(np.random.choice([3, 4, 5], p=[0.25, 0.55, 0.20]))
            banheiros = int(np.random.choice([3, 4, 5], p=[0.25, 0.55, 0.20]))
            vagas = int(np.random.choice([2, 3, 4, 5], p=[0.20, 0.50, 0.20, 0.10]))

        # Fator de idade (depreciação / prêmio de lançamento)
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

        # Fator de vagas de garagem
        fator_vagas = 1.0 + (vagas * 0.05)
        fator_tipo = tipo_mult[tipo]

        # Variação histórica (acumulado 2013-2026)
        taxa_anual = 0.042 if state_uf == "SP" else 0.055  # SC teve valorização acima da média nacional
        fator_ano_transacao = 1.0 + ((ano_transacao - 2013) * taxa_anual)

        preco_m2_estimado = (
            bairro_info["preco_m2_base"]
            * (fator_ano_transacao / (1 + (2026 - 2013) * taxa_anual))
            * fator_idade
            * fator_tipo
            * fator_vagas
        )

        ruido = np.random.normal(1.0, 0.06)
        preco_m2_final = round(preco_m2_estimado * ruido, 2)
        preco_total = round(preco_m2_final * area_m2, 2)

        records.append({
            "id": f"{state_uf}-{start_id + i:06d}",
            "estado": state_uf,
            "cidade": cidade_nome,
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

    return records


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)

    bairros_sp = load_bairros(BAIRROS_SP_FILE)
    bairros_sc = load_bairros(BAIRROS_SC_FILE)

    print(f"[1/3] Gerando dados de São Paulo (SP): {len(bairros_sp)} bairros...")
    records_sp = generate_state_data("SP", "São Paulo", bairros_sp, n_samples=7500, start_id=1, random_seed=42)

    print(f"[2/3] Gerando dados de Santa Catarina (SC): {len(bairros_sc)} bairros/cidades...")
    records_sc = generate_state_data("SC", "Santa Catarina", bairros_sc, n_samples=5000, start_id=1, random_seed=123)

    all_records = records_sp + records_sc
    df = pd.DataFrame(all_records)

    print(f"[3/3] Salvando base unificada em: {OUTPUT_DATASET}")
    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")

    print(f"Dataset concluído com sucesso! Total: {len(df):,} registros.")
    print(f"Distribuição por Estado:\n{df['estado'].value_counts()}\n")
    print("Amostra das cidades de SC inseridas:")
    print(df[df["estado"] == "SC"][["cidade", "bairro", "preco_m2", "preco"]].head(4))


if __name__ == "__main__":
    main()
