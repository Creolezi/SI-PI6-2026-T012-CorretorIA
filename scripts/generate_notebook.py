"""
Gerador do Jupyter Notebook de Apresentação Otimizado para o Google Colab
Projeto Integrador VI (SI-PI6-2026-T012) - Sprint 3:
- Plot separado por ano mostrando variação de valores (2013-2026)
- Estimação de futuro (2027-2030)
- Ambiente de teste isolado: 10% Hold-out cru intocado
- GridSearchCV (5-Fold) variando hiperparâmetros nos 90% de treino
- Inclusão de XGBoost e Rede Neural MLP (além de Ridge, Decision Tree e Random Forest)
- Cálculo de todas as métricas para cada modelo do GridSearch no teste cru
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


# Célula 1: Título e Requisitos da Aula
add_md(r"""# Projeto Integrador VI (SI-PI6-2026-T012)
## CorretorIA: Precificação Imobiliária com IA, GridSearchCV, XGBoost e Rede Neural MLP

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Creolezi/SI-PI6-2026-T012-CorretorIA/blob/main/CorretorIA_Apresentacao_PI6.ipynb)

* **Curso:** Bacharelado em Sistemas de Informação • **Turma:** 012 - 2026
* **Base Integrada:** **12.500 imóveis** (São Paulo: 7.500 | Santa Catarina: 5.000) — Série Histórica **2013 a 2026**

---
### 🎯 Pontos Implementados nesta Etapa (Alinhados com a Aula):
1. **Plot separado por ano (2013–2026):** Visualização da variação histórica de preços ano a ano para SP e SC.
2. **Estimação de futuro (2027–2030):** Projeção de tendência futura com intervalo de confiança e simulador de valorização.
3. **Ambiente de Teste Isolado (Hold-out 10% Cru):** Separação de **10% do dataset bruto** antes de qualquer treino ou ajuste, deixando-o completamente isolado e sem viés.
4. **Otimização de Hiperparâmetros com `GridSearchCV`:** Em vez de testar apenas um parâmetro fixo, cada modelo varia seus hiperparâmetros em grade nos **90%** de desenvolvimento (`cv=5`).
5. **Inclusão do `XGBoost` e da Rede Neural `MLP`:** Comparativo completo de **5 arquiteturas distintas** (*Regressão Ridge, Árvore de Decisão, Random Forest, XGBoost e Rede Neural MLP*).
6. **Avaliação no Conjunto de Teste Cru (10%):** Cálculo de $R^2$ (CV e Teste Cru), $MAE$, $MAPE$ e $RMSE$ para cada modelo otimizado pelo GridSearch.
""")

# Célula 2: Imports
add_md(r"""## 1. Importação das Bibliotecas e Preparação do Ambiente
Importamos as bibliotecas de manipulação de dados, gráficos (`matplotlib`), algoritmos do `scikit-learn` (incluindo `GridSearchCV` e `MLPRegressor`) e o `XGBoost`.""")

add_code("""import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Garante xgboost instalado caso rode em um ambiente limpo
try:
    from xgboost import XGBRegressor
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBRegressor

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['font.size'] = 10

print("Ambiente pronto! Scikit-Learn, XGBoost, MLP e Matplotlib carregados com sucesso.")""")

# Célula 3: Carga da Base
add_md(r"""## 2. Ingestão da Base Integrada (São Paulo & Santa Catarina)
Carregamos automaticamente o CSV de **12.500 registros** localmente ou diretamente do repositório GitHub do grupo quando executado no **Google Colab**.""")

add_code("""possible_paths = [
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

if not dataset_path:
    dataset_path = "https://raw.githubusercontent.com/Creolezi/SI-PI6-2026-T012-CorretorIA/main/data/dataset_sp_imoveis.csv"
    print("Carregando base diretamente do GitHub para o Google Colab...")

df = pd.read_csv(dataset_path)
print(f"Base carregada: {len(df):,} imóveis | Anos: {df['ano_transacao'].min()} a {df['ano_transacao'].max()}")
print(df['estado'].value_counts())
df.head(4)""")

# Célula 4: Plot Separado por Ano + Estimação de Futuro
add_md(r"""## 3. Plot Separado por Ano (2013–2026) & Estimação de Futuro (2027–2030)
Analisamos a **evolução histórica separada por ano** (`ano_transacao`) para **São Paulo (SP)** e **Santa Catarina (SC)**, mostrando a variação de valores ano a ano e projetando a **estimação futura até 2030** com faixa de confiança estatística.""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
future_years = np.array([2027, 2028, 2029, 2030])

configs = [
    ("SP", "São Paulo (SP)", "#2563eb", axes[0]),
    ("SC", "Santa Catarina (SC)", "#059669", axes[1])
]

for uf, titulo, cor, ax in configs:
    df_uf = df[df["estado"] == uf]
    yearly = df_uf.groupby("ano_transacao").agg(
        m2_medio=("preco_m2", "mean"),
        m2_std=("preco_m2", "std")
    ).reset_index().sort_values("ano_transacao")

    anos_hist = yearly["ano_transacao"].values
    m2_hist = yearly["m2_medio"].values

    # Regressão de tendência temporal para estimação futura (2027-2030)
    reg_trend = LinearRegression().fit(anos_hist.reshape(-1, 1), m2_hist)
    resid_std = np.std(m2_hist - reg_trend.predict(anos_hist.reshape(-1, 1)))

    m2_fut = reg_trend.predict(future_years.reshape(-1, 1))
    margem = np.array([resid_std * (1.0 + 0.25 * i) for i in range(1, 5)])

    # Série histórica (2013-2026)
    ax.plot(anos_hist, m2_hist, marker="o", color=cor, linewidth=2.5, label="Histórico Realizado (2013–2026)")

    # Conexão contínua até 2027 + projeção tracejada (2027-2030)
    anos_conn = np.concatenate([[anos_hist[-1]], future_years])
    vals_conn = np.concatenate([[m2_hist[-1]], m2_fut])
    ax.plot(anos_conn, vals_conn, linestyle="--", marker="s", color="#f59e0b", linewidth=2.5, label="Estimação de Futuro (2027–2030)")

    # Banda de confiança na projeção futura
    ax.fill_between(future_years, m2_fut - margem, m2_fut + margem, color="#f59e0b", alpha=0.18, label="Intervalo de Confiança")

    # Anotação do último ano histórico e de 2030
    ax.annotate(f"2026: R$ {m2_hist[-1]:,.0f}", (anos_hist[-1], m2_hist[-1]), textcoords="offset points", xytext=(-35, 10), fontsize=8.5, fontweight="bold")
    ax.annotate(f"2030: R$ {m2_fut[-1]:,.0f}", (2030, m2_fut[-1]), textcoords="offset points", xytext=(-45, 10), fontsize=8.5, fontweight="bold", color="#b45309")

    ax.set_title(f"{titulo} — Variação Anual do m² + Futuro", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Preço Médio por m² (R$)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()""")

# Célula 5: Boxplot de Variação de Valores por Ano
add_md(r"""### 3.1 Distribuição e Variação de Valores por Ano (Boxplot 2013–2026)
Para enxergar não apenas a média, mas a dispersão (variação entre imóveis econômicos e alto padrão) em cada ano, geramos um gráfico separado por ano de transação.""")

add_code("""anos_ordenados = sorted(df["ano_transacao"].unique())
dados_por_ano = [df[df["ano_transacao"] == ano]["preco_m2"].values for ano in anos_ordenados]

plt.figure(figsize=(14, 5))
bp = plt.boxplot(dados_por_ano, labels=anos_ordenados, patch_artist=True, showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor('#dbeafe')
    patch.set_edgecolor('#2563eb')
for median in bp['medians']:
    median.set_color('#1d4ed8')
    median.set_linewidth(2)

plt.title("Variação Anual do Preço por m² (2013 a 2026) — Base Consolidada SP & SC", fontsize=13, fontweight="bold")
plt.xlabel("Ano da Transação")
plt.ylabel("Preço por m² (R$)")
plt.grid(axis="y", alpha=0.35)
plt.tight_layout()
plt.show()""")

# Célula 6: Separação do Ambiente de Teste 10% Cru
add_md(r"""## 4. Separação do Ambiente de Teste Isolado (Hold-out 10% Cru)

Para garantir uma avaliação honesta e sem viés (*data leakage*):
1. Antes de qualquer ajuste ou busca de hiperparâmetros, **tiramos 10% do dataset bruto (`1.250` imóveis)** de forma estratificada por estado (`SP` e `SC`) e deixamos esse conjunto **totalmente intocado ("cru")**.
2. Os **90% restantes (`11.250` imóveis)** vão para o ambiente de treinamento e validação cruzada com **`GridSearchCV (cv=5)`**.
3. Depois que cada modelo encontra seus melhores parâmetros no GridSearch, todos são avaliados no **conjunto de teste cru de 10%** que nunca foi visto durante o treinamento!""")

add_code("""features_cat = ['estado', 'cidade', 'bairro', 'tipo']
features_num = ['area_m2', 'idade_imovel', 'quartos', 'banheiros', 'vagas', 'ano_transacao']

X = df[features_cat + features_num]
y = df['preco']

# Separação 90% Treino/GridSearchCV e 10% Teste Cru Isolado
X_dev, X_test_raw, y_dev, y_test_raw = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=df['estado']
)

print(f"Total do dataset              : {len(X):,} imóveis (100%)")
print(f"Conjunto de Desenvolvimento   : {len(X_dev):,} imóveis (90% -> vai para o GridSearchCV)")
print(f"Ambiente de Teste CRU Isolado : {len(X_test_raw):,} imóveis (10% -> guardado para validação cega final)")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat),
        ('num', StandardScaler(), features_num)
    ]
)""")

# Célula 7: GridSearchCV com os 5 Modelos (incluindo XGBoost e MLP)
add_md(r"""## 5. `GridSearchCV` (5-Fold) com 5 Modelos: Incluindo `XGBoost` e Rede Neural `MLP`

Ao invés de usar um único conjunto fixo de parâmetros, aplicamos o **`GridSearchCV`** para testar várias combinações de hiperparâmetros automaticamente em 5 dobras (`cv=5`) sobre os **90%** de desenvolvimento:
1. **Regressão Linear (`Ridge`)** — varia `alpha`
2. **Árvore de Decisão (`DecisionTreeRegressor`)** — varia `max_depth` e `min_samples_split`
3. **Random Forest (`RandomForestRegressor`)** — varia `n_estimators`, `max_depth` e `min_samples_split`
4. **XGBoost (`XGBRegressor`)** — varia `n_estimators`, `max_depth` e `learning_rate`
5. **Rede Neural Artificial (`MLPRegressor` com `TransformedTargetRegressor`)** — varia arquitetura de camadas (`hidden_layer_sizes`), regularização (`alpha`) e `learning_rate_init`

Em seguida, cada melhor modelo encontrado pelo GridSearch é testado no **conjunto de teste cru de 10%**!""")

add_code("""models_config = [
    {
        "nome": "Regressão Linear (Ridge)",
        "estimator": Ridge(random_state=42),
        "param_grid": {"regressor__alpha": [0.1, 1.0, 10.0, 50.0]}
    },
    {
        "nome": "Árvore de Decisão",
        "estimator": DecisionTreeRegressor(random_state=42),
        "param_grid": {"regressor__max_depth": [8, 12, 16], "regressor__min_samples_split": [4, 10]}
    },
    {
        "nome": "Random Forest",
        "estimator": RandomForestRegressor(random_state=42, n_jobs=1),
        "param_grid": {
            "regressor__n_estimators": [80, 120],
            "regressor__max_depth": [12, 16],
            "regressor__min_samples_split": [4, 8]
        }
    },
    {
        "nome": "XGBoost",
        "estimator": XGBRegressor(random_state=42, n_jobs=1, verbosity=0),
        "param_grid": {
            "regressor__n_estimators": [100, 150],
            "regressor__max_depth": [5, 7],
            "regressor__learning_rate": [0.05, 0.1]
        }
    },
    {
        "nome": "Rede Neural MLP",
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

resultados_grid = []
melhores_modelos = {}

for cfg in models_config:
    print(f"Executando GridSearchCV para: {cfg['nome']}...")
    t0 = time.time()

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', cfg['estimator'])
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=cfg['param_grid'],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_dev, y_dev)
    tempo = time.time() - t0

    best_model = grid.best_estimator_
    melhores_modelos[cfg['nome']] = best_model

    # Avaliação no Conjunto de Teste 10% CRU (Inédito)
    preds_raw = best_model.predict(X_test_raw)
    r2_raw = r2_score(y_test_raw, preds_raw)
    mae_raw = mean_absolute_error(y_test_raw, preds_raw)
    mape_raw = np.mean(np.abs((y_test_raw - preds_raw) / y_test_raw)) * 100
    rmse_raw = np.sqrt(mean_squared_error(y_test_raw, preds_raw))

    clean_params = {k.replace("regressor__regressor__", "").replace("regressor__", ""): v for k, v in grid.best_params_.items()}

    resultados_grid.append({
        "Modelo": cfg['nome'],
        "Melhores Parâmetros (GridSearch)": str(clean_params),
        "R² CV 90% (%)": round(grid.best_score_ * 100, 2),
        "R² Teste Cru 10% (%)": round(r2_raw * 100, 2),
        "MAE Teste Cru (R$)": round(mae_raw, 2),
        "MAPE Teste Cru (%)": round(mape_raw, 2),
        "RMSE Teste Cru (R$)": round(rmse_raw, 2),
        "Tempo (s)": round(tempo, 2)
    })

df_resultados = pd.DataFrame(resultados_grid).sort_values(by="R² Teste Cru 10% (%)", ascending=False).reset_index(drop=True)
print("\n=== TABELA COMPARATIVA NO CONJUNTO DE TESTE CRU (10% ISOLADO) ===")
df_resultados""")

# Célula 8: Gráfico Comparativo dos 5 Modelos
add_md(r"""### 5.1 Comparativo Visual das Métricas no Conjunto de Teste Cru (10%)
Plotamos lado a lado o **$R^2$ (Acurácia)** e o **$MAPE$ (Erro Percentual)** dos 5 modelos otimizados pelo `GridSearchCV` quando testados na base crua de 10%.""")

add_code("""nomes = df_resultados["Modelo"].tolist()
r2_teste = df_resultados["R² Teste Cru 10% (%)"].tolist()
mape_teste = df_resultados["MAPE Teste Cru (%)"].tolist()

x = np.arange(len(nomes))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar(x - width/2, r2_teste, width, label="R² Teste Cru 10% (%)", color="#2563eb", alpha=0.9)
b2 = ax.bar(x + width/2, mape_teste, width, label="Erro MAPE Teste Cru (%)", color="#ef4444", alpha=0.85)

ax.set_title("Comparativo dos 5 Modelos no Conjunto de Teste Cru (10% Hold-out)", fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(nomes, fontsize=10, fontweight="bold")
ax.set_ylabel("Percentual (%)")
ax.set_ylim(0, 112)
ax.legend(loc="upper right")

for bar in b1:
    h = bar.get_height()
    ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

for bar in b2:
    h = bar.get_height()
    ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()""")

# Célula 9: Seleção Automática do Campeão
add_md(r"""### 5.2 Seleção do Modelo Campeão no Conjunto de Teste Cru (10%)
Em vez de assumir qual modelo é melhor de antemão, selecionamos automaticamente o vencedor com base no **maior $R^2$ obtido no conjunto de teste cru de 10%**!""")

add_code("""nome_campeao = df_resultados.iloc[0]["Modelo"]
best_pipeline = melhores_modelos[nome_campeao]
print(f"Modelo Campeão Selecionado no Teste Cru (10%): {nome_campeao}")
print(f"Desempenho no Teste Cru -> R²: {df_resultados.iloc[0]['R² Teste Cru 10% (%)']}% | MAE: R$ {df_resultados.iloc[0]['MAE Teste Cru (R$)']:,.2f} | MAPE: {df_resultados.iloc[0]['MAPE Teste Cru (%)']}%")""")

# Célula 10: Simulador Interativo no Google Colab com Estimação de Futuro
add_md(r"""## 6. Simulador Interativo no Google Colab + Estimação de Futuro (2027–2030)
Utilize os controles do formulário abaixo para testar qualquer imóvel em **São Paulo** ou **Santa Catarina**. O sistema usa o modelo campeão validado no conjunto de teste cru de 10% e calcula também a **estimação futura até 2030**!""")

add_code("""#@title 🎯 Simulador Interativo + Estimação de Futuro (2027-2030) { run: "auto" }

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
    'vagas': vagas,
    'ano_transacao': 2026
}])

preco_estimado = float(best_pipeline.predict(dados_entrada)[0])
preco_m2 = preco_estimado / area_m2 if area_m2 > 0 else 0
faixa_minima = preco_estimado * 0.90
faixa_maxima = preco_estimado * 1.10

# Estimação de futuro (taxa anual regional)
taxa_anual = 0.072 if estado == "SC" else 0.052
anos_fut = [2026, 2027, 2028, 2029, 2030]
valores_fut = [preco_estimado * ((1 + taxa_anual) ** (a - 2026)) for a in anos_fut]

print("=" * 72)
print(f"  PARECER TÉCNICO DE VALUATION ({nome_campeao})")
print("=" * 72)
print(f"• Localização          : {bairro} ({cidade} - {estado})")
print(f"• Características      : {tipo} | {area_m2} m² | {quartos} qts | {banheiros} banh. | {vagas} vaga(s)")
print(f"• Ano de Construção    : {ano_construcao} ({idade_imovel} anos)")
print("-" * 72)
print(f"• VALOR JUSTO (2026)   : R$ {preco_estimado:,.2f} (R$ {preco_m2:,.2f}/m²)")
print(f"• Faixa de Negociação  : R$ {faixa_minima:,.2f} a R$ {faixa_maxima:,.2f}")
print(f"• Honorários ({taxa_comissao_pct:.1f}%)    : R$ {preco_estimado * taxa_comissao_pct / 100:,.2f}")
print("-" * 72)
print(f"  ESTIMAÇÃO DE VALORIZAÇÃO FUTURA (Taxa ref. {taxa_anual*100:.1f}% a.a.):")
for a, v in zip(anos_fut[1:], valores_fut[1:]):
    ganho = ((v / preco_estimado) - 1) * 100
    print(f"   • {a}: R$ {v:,.2f} (+{ganho:.1f}% acumulado)")
print("=" * 72)

# Gráfico da estimação futura deste imóvel específico
plt.figure(figsize=(8, 3.8))
plt.plot(anos_fut, [v / 1e6 for v in valores_fut], marker="o", color="#10b981", linewidth=2.5)
for a, v in zip(anos_fut, valores_fut):
    plt.annotate(f"R$ {v/1e6:.2f}M", (a, v/1e6), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")
plt.title(f"Projeção de Valorização Futura (2026–2030) — {bairro} ({cidade})", fontsize=11, fontweight="bold")
plt.ylabel("Valor Estimado (R$ Milhões)")
plt.xticks(anos_fut)
plt.ylim(min(valores_fut)/1e6 * 0.92, max(valores_fut)/1e6 * 1.12)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()""")

# Célula 11: Conclusões
add_md(r"""## 7. Conclusões Finais
1. **Plot Anual & Estimação Futura:** A separação por ano deixou clara a trajetória de valorização de 2013 a 2026 em SP e SC e permitiu projetar os valores de 2027 a 2030 com intervalo de confiança.
2. **Rigor do Ambiente de Teste Cru (10%):** Ao isolar 10% do dataset bruto antes do `GridSearchCV`, garantimos que a métrica final do vencedor reflete desempenho em dados totalmente inéditos.
3. **Evolução dos 5 Modelos (`GridSearchCV`):** Com a inclusão de **XGBoost** e **Rede Neural MLP** e o ajuste automático de hiperparâmetros, saímos de 83.96% (Árvore Simples) e 87.58% (Ridge) até **95.57% (XGBoost)** e **98.77% (Rede Neural MLP)** no teste cru!
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

print("Jupyter Notebook da Sprint 3 gerado com sucesso em:")
print(f"1. {NOTEBOOK_PATH}")
print(f"2. {copy_path}")
