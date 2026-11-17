"""
Script de Preparação e Geração de Dataset Imobiliário Multi-Estado (SP, SC, MG, RJ, DF, PR)
Projeto Integrador 6 (SI-PI6-2026-T012)

Gera e unifica bases imobiliárias de 6 estados brasileiros (18.000 registros)
cobrindo a série histórica de 2013 a 2026 com base em parâmetros de mercado (FipeZAP/ITBI).
"""

import os
import json
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DATASET = os.path.join(DATA_DIR, "dataset_sp_imoveis.csv")

ESTADOS_CONFIG = [
    {"uf": "SP", "nome": "São Paulo", "file": "bairros_sp.json", "n_samples": 5000, "seed": 42},
    {"uf": "SC", "nome": "Santa Catarina", "file": "bairros_sc.json", "n_samples": 3500, "seed": 123},
    {"uf": "RJ", "nome": "Rio de Janeiro", "file": "bairros_rj.json", "n_samples": 3000, "seed": 456},
    {"uf": "MG", "nome": "Minas Gerais", "file": "bairros_mg.json", "n_samples": 2500, "seed": 789},
    {"uf": "PR", "nome": "Paraná", "file": "bairros_pr.json", "n_samples": 2000, "seed": 1011},
    {"uf": "DF", "nome": "Distrito Federal", "file": "bairros_df.json", "n_samples": 2000, "seed": 1213},
]


def load_bairros(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_state_data(state_uf, state_name, bairros_list, n_samples, start_id, random_seed):
    np.random.seed(random_seed)
    bairros_dict = {b["nome"]: b for b in bairros_list}
    nomes_bairros = list(bairros_dict.keys())

    tipos = ["Apartamento", "Casa", "Cobertura"]
    if state_uf in ("SC", "RJ"):
        tipo_prob = [0.74, 0.18, 0.08]
    elif state_uf == "DF":
        tipo_prob = [0.70, 0.24, 0.06]
    else:
        tipo_prob = [0.73, 0.22, 0.05]

    tipo_mult = {"Apartamento": 1.0, "Casa": 0.92, "Cobertura": 1.45}
    records = []

    for i in range(n_samples):
        bairro_nome = np.random.choice(nomes_bairros)
        bairro_info = bairros_dict[bairro_nome]
        cidade_nome = bairro_info.get("cidade", state_name)
        tipo = np.random.choice(tipos, p=tipo_prob)

        # Ano de construção (1975 até 2025)
        ano_construcao = int(np.random.choice(
            np.arange(1975, 2026),
            p=np.linspace(0.005, 0.035, 51) / np.sum(np.linspace(0.005, 0.035, 51))
        ))
        idade_imovel = 2026 - ano_construcao

        # Ano de transação/anúncio (2013 a 2026)
        ano_transacao = int(np.random.randint(2013, 2027))

        # Metragem quadrada
        if tipo == "Apartamento":
            area_m2 = int(np.clip(np.random.exponential(scale=48) + 35, 30, 420))
        elif tipo == "Casa":
            area_m2 = int(np.clip(np.random.exponential(scale=75) + 85, 60, 650))
        else:
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

        # Fator de idade
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

        fator_vagas = 1.0 + (vagas * 0.05)
        fator_tipo = tipo_mult[tipo]

        # Usa a taxa anual do próprio bairro para refletir a variação histórica fiel
        taxa_anual = float(bairro_info.get("valorizacao_anual", 0.05))
        fator_ano_transacao = 1.0 + ((ano_transacao - 2013) * taxa_anual)

        preco_m2_estimado = (
            bairro_info["preco_m2_base"]
            * (fator_ano_transacao / (1 + (2026 - 2013) * taxa_anual))
            * fator_idade
            * fator_tipo
            * fator_vagas
        )

        ruido = np.random.normal(1.0, 0.055)
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

    all_records = []
    for idx, cfg in enumerate(ESTADOS_CONFIG, start=1):
        filepath = os.path.join(DATA_DIR, cfg["file"])
        bairros = load_bairros(filepath)
        print(f"[{idx}/{len(ESTADOS_CONFIG)}] Gerando {cfg['n_samples']:,} amostras para {cfg['nome']} ({cfg['uf']}) ({len(bairros)} bairros)...")
        recs = generate_state_data(
            cfg["uf"], cfg["nome"], bairros,
            n_samples=cfg["n_samples"], start_id=1, random_seed=cfg["seed"]
        )
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    print(f"Salvando base unificada de {len(df):,} registros em: {OUTPUT_DATASET}")
    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")

    print("\nResumo por Estado:")
    print(df["estado"].value_counts())


if __name__ == "__main__":
    main()
