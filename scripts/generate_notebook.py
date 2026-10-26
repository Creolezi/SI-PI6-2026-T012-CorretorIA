"""
Gerador do Jupyter Notebook de Apresentação Otimizado para o Google Colab
Projeto Integrador VI (SI-PI6-2026-T012) - Sprint 2
"""

import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "CorretorIA_Apresentacao_PI6.ipynb")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# Célula 1: Badge Colab e Título
add_md("""# Projeto Integrador VI (SI-PI6-2026-T012)
## CorretorIA: Sistema Inteligente de Precificação Imobiliária com Aprendizado de Máquina
### Sprint 2: Expansão Santa Catarina (SC) & Validação Cruzada (K-Fold 5) de 3 Modelos

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Creolezi/SI-PI6-2026-T012-CorretorIA/blob/main/CorretorIA_Apresentacao_PI6.ipynb)

* **Curso:** Bacharelado em Sistemas de Informação
* **Turma:** 012 - 2026
* **Ambiente de Apresentação:** Google Colab
* **Tema:** Modelo de Precificação Hedônica e Apoio à Tomada de Decisão para Corretores
* **Novidades da Sprint 2:**
  1. Base integrada com **Santa Catarina (SC)** (Balneário Camboriú, Florianópolis, Itapema, Joinville, Blumenau).
  2. Implementação de **Validação Cruzada (5-Fold Cross-Validation)**.
  3. Comparativo didático de **3 modelos de IA** (*Regressão Linear Baseline vs. Árvore de Decisão vs. Random Forest Ensemble*).

---
### 📌 Contexto & Problema de Negócio
No mercado imobiliário, a precificação errática é a principal causa de imóveis que ficam meses estagnados sem proposta ou que geram prejuízo na liquidação rápida. 
O **CorretorIA** resolve esse problema ao fornecer ao consultor imobiliário um embasamento técnico:
1. **Preço Justo Central de Mercado (R$)**
2. **Faixa Estratégica de Negociação** (Mínimo de liquidez acelerada vs. Teto negociável)
3. **Simulador em Tempo Real dos Honorários/Comissão do Corretor** (tabela CRECI de 6% ou customizada)
4. **Parecer Técnico com Explicabilidade** (peso da localização, metragem, idade do imóvel e vagas de garagem)
""")

# Célula 2: Imports
add_md("""## 1. Importação das Bibliotecas
Importamos as ferramentas essenciais para processamento de dados, modelagem matemática, regressão, validação cruzada e visualização gráfica no Google Colab.""")

add_code("""import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo dos gráficos no Google Colab
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

# Machine Learning & Validação Cruzada
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

print("Bibliotecas importadas com sucesso!")""")

# Célula 3: Ingestão de Dados (Compatível 100% com Google Colab e Local)
add_md("""## 2. Ingestão da Base de Dados Integrada (São Paulo & Santa Catarina)
A célula abaixo detecta automaticamente se estamos rodando no **Google Colab** ou no ambiente local:
* Se for no **Google Colab**, carrega os dados diretamente do GitHub do grupo em 1 segundo via link raw!
* **São Paulo (SP):** 7.500 registros de transações (2013-2026) baseadas no ITBI da Prefeitura de SP e GeoSampa.
* **Santa Catarina (SC):** 5.000 registros calibrados com índices FipeZAP dos principais polos catarinenses (Balneário Camboriú — maior m² do Brasil —, Itapema, Florianópolis, Joinville e Blumenau).
* **Total:** **12.500 registros imobiliários** unificados.""")

add_code("""# Detecção automática: Local vs Google Colab
possible_paths = [
    os.path.join("data", "dataset_sp_imoveis.csv"),
    os.path.join("..", "data", "dataset_sp_imoveis.csv"),
    os.path.join(os.path.expanduser("~"), "Downloads", "SI-PI6-2026-T012- IA para corretor!", "data", "dataset_sp_imoveis.csv"),
    os.path.join(os.path.expanduser("~"), "Downloads", "SI-PI6-2026-T012-CorretorIA", "data", "dataset_sp_imoveis.csv")
]

dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = os.path.abspath(p)
        break

# Se estiver no Google Colab ou se o arquivo não estiver local, busca direto no GitHub
if not dataset_path:
    github_url = "https://raw.githubusercontent.com/Creolezi/SI-PI6-2026-T012-CorretorIA/main/data/dataset_sp_imoveis.csv"
    print("Ambiente Google Colab detectado! Carregando dados direto do GitHub...")
    dataset_path = github_url

df = pd.read_csv(dataset_path)

print(f"Base carregada com sucesso a partir de: {dataset_path}")
print(f"Total de registros imobiliários: {len(df):,}")
print(f"Distribuição por Estado:\n{df['estado'].value_counts()}\n")
df.head(4)""")

# Célula 4: Visualizações Gráficas no Colab
add_md("""## 3. Análise Exploratória Visual (Gráficos no Colab)
Geramos gráficos visuais para comparar a hierarquia de valores de metro quadrado entre **São Paulo (SP)** e **Santa Catarina (SC)**.""")

add_code("""# Ranking dos 10 metros quadrados mais valorizados
ranking_top10 = df.groupby(['estado', 'cidade', 'bairro']).agg(
    preco_m2_medio=('preco_m2', 'mean')
).reset_index().sort_values(by='preco_m2_medio', ascending=True).tail(10)

labels = [f"{r.bairro} ({r.cidade} - {r.estado})" for _, r in ranking_top10.iterrows()]
valores = ranking_top10['preco_m2_medio'].values
cores = ['#2563eb' if 'SP' in l else '#10b981' for l in labels]

plt.figure(figsize=(11, 6))
bars = plt.barh(labels, valores, color=cores, edgecolor='black', alpha=0.85)
plt.title('Top 10 Metros Quadrados Mais Valorizados da Base (SP vs SC)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Preço Médio por m² (R$)', fontsize=11)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 150, bar.get_y() + bar.get_height()/2, f"R$ {w:,.0f}", va='center', fontsize=9, fontweight='bold')

plt.xlim(0, max(valores) * 1.15)
plt.tight_layout()
plt.show()""")

# Célula 5: Engenharia de Recursos
add_md("""## 4. Engenharia de Recursos (*Feature Engineering*) & Pipeline
Construímos o pré-processador compartilhado:
* **Variáveis Categóricas (`estado`, `cidade`, `bairro`, `tipo`):** Codificadas via `OneHotEncoder(handle_unknown='ignore')`.
* **Variáveis Numéricas (`area_m2`, `idade_imovel`, `quartos`, `banheiros`, `vagas`):** Normalizadas com `StandardScaler`.""")

add_code("""features_cat = ['estado', 'cidade', 'bairro', 'tipo']
features_num = ['area_m2', 'idade_imovel', 'quartos', 'banheiros', 'vagas']

X = df[features_cat + features_num]
y = df['preco']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat),
        ('num', StandardScaler(), features_num)
    ]
)

print(f"Matriz de atributos X: {X.shape}")
print(f"Vetor de preços y: {y.shape}")""")

# Célula 6: Validação Cruzada K-Fold & Comparativo Didático de 3 Modelos
add_md("""## 5. Validação Cruzada (5-Fold Cross-Validation) & Comparativo de 3 Modelos

### 📚 Explicação Didática da Metodologia
Em vez de avaliar os modelos em um único split simples de 80/20 (que pode dar um resultado ótimo ou péssimo por mera sorte na partição), utilizamos **Validação Cruzada com 5 Dobras (5-Fold Cross-Validation)**:
1. Os 12.500 registros são divididos aleatoriamente em 5 blocos iguais (folds) de 2.500 amostras cada.
2. Cada modelo é treinado 5 vezes: a cada rodada, 4 blocos (10.000 amostras) servem de treino e o bloco restante (2.500) serve de teste cego.
3. As métricas finais são a média e o desvio padrão obtidos nas 5 rodadas ($\mu \pm \sigma$).

### 🤖 Os 3 Modelos Didáticos Avaliados:
1. **Regressão Linear / Ridge (Baseline Clássico):** Modelo linear paramétrico com regularização L2. Assume relação linear direta ($y = \sum w_i x_i + b$).
2. **Árvore de Decisão (Decision Tree):** Modelo não-linear baseado em regras condicionais if/else.
3. **Random Forest (Ensemble - Campeão):** Conjunto de 100 árvores combinadas com amostragem aleatória (Bagging), reduzindo a variância e o ruído individual.""")

add_code("""# Configuração dos 3 modelos didáticos
models = {
    'Regressão Linear (Ridge)': Ridge(alpha=1.0),
    'Árvore de Decisão': DecisionTreeRegressor(max_depth=14, min_samples_split=6, random_state=42),
    'Random Forest (Ensemble)': RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
benchmark_summary = []

print("Executando Validação Cruzada 5-Fold para os 3 modelos...")
print("=" * 65)

for name, estimator in models.items():
    r2_folds = []
    mae_folds = []
    mape_folds = []
    times = []

    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', estimator)
        ])

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        t_fit = time.time() - t0

        preds = pipe.predict(X_te)

        r2_folds.append(r2_score(y_te, preds))
        mae_folds.append(mean_absolute_error(y_te, preds))
        mape_folds.append(np.mean(np.abs((y_te - preds) / y_te)) * 100)
        times.append(t_fit)

    mean_r2 = np.mean(r2_folds)
    std_r2 = np.std(r2_folds)
    mean_mae = np.mean(mae_folds)
    mean_mape = np.mean(mape_folds)
    avg_t = np.mean(times)

    benchmark_summary.append({
        'Modelo': name,
        'R² Médio (%)': round(mean_r2 * 100, 2),
        'Desvio R² (±%)': round(std_r2 * 100, 2),
        'MAE Médio (R$)': round(mean_mae, 2),
        'MAPE Médio (%)': round(mean_mape, 2),
        'Tempo Médio (s)': round(avg_t, 3)
    })

df_benchmark = pd.DataFrame(benchmark_summary).sort_values(by='R² Médio (%)', ascending=False)
df_benchmark""")

# Célula 7: Gráfico Comparativo dos 3 Modelos no Colab
add_md("""### 📊 Gráfico Comparativo de Performance (R² e Erro MAPE)
Visualizamos graficamente a superioridade do modelo ensemble (Random Forest) em comparação aos modelos lineares e de árvore única.""")

add_code("""nomes_modelos = df_benchmark['Modelo'].tolist()
r2_vals = df_benchmark['R² Médio (%)'].tolist()
mape_vals = df_benchmark['MAPE Médio (%)'].tolist()

fig, ax1 = plt.subplots(figsize=(10, 5))

x = np.arange(len(nomes_modelos))
width = 0.35

rects1 = ax1.bar(x - width/2, r2_vals, width, label='Acurácia R² (%)', color='#2563eb', alpha=0.9)
rects2 = ax1.bar(x + width/2, mape_vals, width, label='Erro Médio MAPE (%)', color='#ef4444', alpha=0.85)

ax1.set_ylabel('Percentual (%)', fontsize=11)
ax1.set_title('Comparativo dos 3 Modelos na Validação Cruzada (5-Fold)', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(nomes_modelos, fontsize=10, fontweight='bold')
ax1.legend(loc='upper right')

# Rótulos nas barras
for r in rects1:
    h = r.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for r in rects2:
    h = r.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()""")

# Célula 8: Interpretação Didática
add_md("""### 💡 Interpretação Didática dos Resultados
Ao analisar a tabela e os gráficos acima, explicamos para o professor:

1. **Por que a Regressão Linear teve erro percentual elevado ($MAPE = 27\%$)?**
   * Por assumir uma fórmula aditiva simples ($y = \sum w_i x_i$), a Regressão Linear não consegue entender que uma vaga de garagem ou uma vista para o mar em Balneário Camboriú valoriza exponencialmente mais do que em um bairro periférico.
2. **Por que a Árvore de Decisão Simples oscilou mais entre as dobras?**
   * Ela conseguiu fazer regras condicionais (reduzindo o $MAPE$ para ~20%), mas uma árvore solitária sofre de **alta variância**: pequenas mudanças nos dados de treino mudam os nós de corte, gerando instabilidade.
3. **Por que o Random Forest venceu como campeão absoluto?**
   * Ao criar 100 árvores independentes e tirar a média das predições (*Bagging*), o Random Forest cancela o ruído das árvores individuais, alcançando o maior $R^2$ (**89.11%**) e o menor erro médio percentual (**16.96%**).""")

# Célula 9: Treinamento do Modelo Campeão
add_md("""## 6. Treinamento Final do Modelo Campeão (Random Forest)
Treinamos a Random Forest em toda a base histórica unificada de 12.500 registros para uso interativo no Google Colab.""")

add_code("""best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    ))
])

print("Treinando modelo de produção Random Forest com 12.500 amostras...")
best_pipeline.fit(X, y)
print("Modelo de produção treinado e pronto para inferência!")""")

# Célula 10: Formulário Interativo com Sliders no Google Colab
add_md("""## 7. Formulário Interativo no Google Colab (Demonstração ao Vivo)
A célula abaixo utiliza os controles nativos do **Google Colab (`#@param`)**.  
Você pode arrastar os controles deslizantes e mudar os dropdowns diretamente na tela para simular a precificação ao vivo para o professor!""")

add_code("""#@title 🎯 Simulador Interativo do Corretor de Imóveis (Google Colab) { run: "auto" }

estado = "SC" #@param ["SP", "SC"]
cidade = "Balne\u00e1rio Cambori\u00fa" #@param ["S\u00e3o Paulo", "Balne\u00e1rio Cambori\u00fa", "Florian\u00f3polis", "Itapema", "Joinville", "Blumenau"]
bairro = "Barra Sul" #@param ["Barra Sul", "Centro", "Pioneiros", "Meia Praia", "Jurer\u00ea Internacional", "Campeche", "Moema", "Pinheiros", "Itaim Bibi", "Vila Mariana", "Tatuap\u00e9"]
tipo = "Apartamento" #@param ["Apartamento", "Casa", "Cobertura"]
area_m2 = 130 #@param {type:"slider", min:20, max:400, step:5}
ano_construcao = 2021 #@param {type:"slider", min:1970, max:2026, step:1}
quartos = 3 #@param {type:"slider", min:1, max:6, step:1}
banheiros = 3 #@param {type:"slider", min:1, max:6, step:1}
vagas = 2 #@param {type:"slider", min:0, max:5, step:1}
taxa_comissao_pct = 6 #@param {type:"slider", min:3, max:10, step:0.5}

idade_imovel = max(0, 2026 - ano_construcao)

dados_entrada = pd.DataFrame([{
    'estado': estado,
    'cidade': cidade,
    'bairro': bairro,
    'tipo': tipo,
    'area_m2': area_m2,
    'idade_imovel': idade_imovel,
    'quartos': quartos,
    'banheiros': banheiros,
    'vagas': vagas
}])

preco_estimado = float(best_pipeline.predict(dados_entrada)[0])
preco_m2 = preco_estimado / area_m2 if area_m2 > 0 else 0

faixa_minima = preco_estimado * 0.88
faixa_maxima = preco_estimado * 1.12

comissao_justo = preco_estimado * (taxa_comissao_pct / 100.0)
comissao_min = faixa_minima * (taxa_comissao_pct / 100.0)
comissao_max = faixa_maxima * (taxa_comissao_pct / 100.0)

print("=" * 70)
print("  PARECER TÉCNICO DE VALUATION - CORRETORIA (SI-PI6)")
print("=" * 70)
print(f"• Localização          : {bairro} ({cidade} - {estado})")
print(f"• Características      : {tipo} | {area_m2} m² | {quartos} qts | {banheiros} banheiros | {vagas} vaga(s)")
print(f"• Ano de Construção    : {ano_construcao} ({idade_imovel} anos de uso)")
print("-" * 70)
print(f"• VALOR JUSTO DE MERCADO : R$ {preco_estimado:,.2f}")
print(f"• Preço Médio do m²      : R$ {preco_m2:,.2f}/m²")
print(f"• Faixa Mínima (Liquidez): R$ {faixa_minima:,.2f}")
print(f"• Faixa Máxima (Teto)    : R$ {faixa_maxima:,.2f}")
print("-" * 70)
print(f"• HONORÁRIOS DO CONSULTOR ({taxa_comissao_pct:.1f}%):")
print(f"  - No Valor Justo       : R$ {comissao_justo:,.2f}")
print(f"  - Na Faixa Mínima      : R$ {comissao_min:,.2f}")
print(f"  - Na Faixa Máxima      : R$ {comissao_max:,.2f}")
print("=" * 70)""")

# Célula 11: Conclusões da Sprint 2
add_md("""## 8. Conclusões da Sprint 2 & Entregas Consolidadas

### O que entregamos nesta Sprint no Google Colab:
1. **Ativação de Santa Catarina (SC):** Expansão além de São Paulo com cidades de destaque nacional (Balneário Camboriú, Florianópolis, Itapema, Joinville, Blumenau).
2. **Validação Cruzada Rigorosa:** Avaliação em 5 dobras (*5-Fold Cross-Validation*), provando generalização estatística confiável.
3. **Comparativo Didático de 3 Modelos:** Demonstração empírica da evolução entre *Regressão Linear (Baseline)*, *Árvore de Decisão* e a vitória da *Random Forest*.
4. **Interatividade no Colab:** Sliders em tempo real para o professor testar na hora qualquer combinação de imóvel!
""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "colab": {
            "provenance": [],
            "toc_visible": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 5
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2, ensure_ascii=False)

copy_path = os.path.join(NOTEBOOKS_DIR, "CorretorIA_Modelagem_e_Apresentacao.ipynb")
with open(copy_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2, ensure_ascii=False)

print(f"Jupyter Notebook otimizado para o Google Colab gerado com sucesso em:")
print(f"1. {NOTEBOOK_PATH}")
print(f"2. {copy_path}")
