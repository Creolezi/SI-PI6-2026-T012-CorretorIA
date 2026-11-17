# CorretorIA - Sistema de Precificação Imobiliária com Inteligência Artificial

> Projeto Integrador VI (SI-PI6-2026-T012)  
> Curso: Sistemas de Informação • Sprint 3 (GridSearchCV, Teste Cru 10%, XGBoost, Rede Neural MLP & Estimação Futura)

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Creolezi/SI-PI6-2026-T012-CorretorIA/blob/main/CorretorIA_Apresentacao_PI6.ipynb)

---

## 📌 Sobre o Projeto

A gente pensou nesse projeto para resolver uma dor real que quase todo corretor e consultor imobiliário enfrenta no dia a dia: **como precificar um imóvel com segurança e rapidez sem cair na ilusão do proprietário ou desvalorizar o imóvel.**

O nosso sistema, batizado de **CorretorIA**, ajuda o consultor imobiliário a avaliar imóveis de **São Paulo (SP)** e **Santa Catarina (SC)** com embasamento estatístico, entregando:
1. **Preço Justo Central:** Valor de mercado estimado pelo modelo campeão.
2. **Faixa de Negociação:** Margem mínima (liquidez rápida) e teto negociável.
3. **Estimação de Futuro (2027–2030):** Projeção de quanto o imóvel tende a valorizar nos próximos 4 anos.
4. **Simulador de Comissão:** Cálculo direto dos honorários do consultor (6% padrão CRECI ou personalizado).
5. **Parecer Técnico Explicativo:** Fatores que mais influenciaram o valor do imóvel.

---

## 🗄️ De Onde Vieram os Nossos Datasets? (SP & SC)

Unificamos uma base histórica com **12.500 registros de transações (2013 até 2026)**:
* **São Paulo (SP - 7.500 registros):** Calibrados nos padrões públicos do ITBI da Prefeitura de SP, GeoSampa e índices de mercado (FipeZAP).
* **Santa Catarina (SC - 5.000 registros):** Principais polos do estado (Balneário Camboriú — Barra Sul, Centro e Pioneiros —, Itapema — Meia Praia e Centro —, Florianópolis — Jurerê Internacional, Campeche, Agronômica, Centro e Itacorubi —, Joinville e Blumenau).

---

## 📈 Variação Anual (2013–2026) & Estimação de Futuro (2027–2030)

Para atender às discussões da última aula:
1. **Plot separado por ano (`ano_transacao`):** Separamos as transações de **2013 até 2026** para visualizar claramente como o preço médio e a variação do m² evoluíram ano a ano em São Paulo e em Santa Catarina.
2. **Estimação de Futuro (2027–2030):** Ajustamos uma tendência temporal sobre a série histórica anual de cada estado (com faixa de confiança) e aplicamos a taxa de valorização regional no simulador para estimar o valor futuro do imóvel em **2027, 2028, 2029 e 2030**.

---

## 🧪 Ambiente de Teste Isolado (Hold-out 10% Cru) & `GridSearchCV`

Na etapa anterior avaliávamos os modelos diretamente com validação cruzada e parâmetros fixos. Nesta entrega evoluímos o rigor metodológico:

1. **Separação de 10% do Dataset "Cru" (`Hold-out`):**
   * Antes de qualquer ajuste ou busca de hiperparâmetros, **tiramos 10% do dataset bruto (1.250 imóveis)** de forma estratificada por estado (`SP` e `SC`) e deixamos esse bloco totalmente de lado.
   * Assim, o modelo vencedor nunca teve contato com esses dados, garantindo uma avaliação cega sem nenhum viés.
2. **Busca de Hiperparâmetros com `GridSearchCV` (5-Fold nos 90% restantes):**
   * Em vez de testar apenas um conjunto fixo de parâmetros, usamos os **90% restantes (11.250 imóveis)** dentro do `GridSearchCV(cv=5)` para variar os hiperparâmetros em grade de todos os modelos e escolher automaticamente a melhor configuração de cada um.

---

## 🤖 Comparativo dos 5 Modelos (incluindo `XGBoost` e `Rede Neural MLP`)

Todas as métricas abaixo foram calculadas para cada modelo otimizado pelo `GridSearchCV` e avaliadas no **conjunto de teste cru de 10% (1.250 imóveis inéditos)**:

| Modelo de IA | Melhores Parâmetros (`GridSearchCV`) | R² CV (90%) | R² Teste Cru (10%) | MAE (Teste Cru) | MAPE (Teste Cru) | RMSE (Teste Cru) | Tempo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rede Neural MLP** 🏆 | `{'alpha': 0.001, 'hidden_layer_sizes': [64, 32], 'learning_rate_init': 0.005}` | **98.61%** | **98.77%** | **R$ 75.144,29** | **6.24%** | **R$ 132.956,46** | 15.50s |
| **XGBoost** | `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150}` | 96.45% | 95.57% | R$ 94.471,28 | 7.02% | R$ 252.241,68 | 4.98s |
| **Random Forest** | `{'max_depth': 16, 'min_samples_split': 4, 'n_estimators': 120}` | 91.59% | 91.68% | R$ 159.587,28 | 12.41% | R$ 345.499,69 | 29.55s |
| **Regressão Linear (Ridge)** | `{'alpha': 1.0}` | 89.02% | 87.58% | R$ 220.306,01 | 26.17% | R$ 422.186,50 | 4.05s |
| **Árvore de Decisão** | `{'max_depth': 16, 'min_samples_split': 10}` | 83.87% | 83.96% | R$ 215.849,67 | 16.30% | R$ 479.909,58 | 0.69s |

### O que a gente concluiu comparando os 5 modelos:
* **Árvore de Decisão & Regressão Linear (Ridge):** São rápidos, mas erram mais nas combinações de bairros nobres, metragem e ano da transação (`MAPE` de 16.30% e 26.17%).
* **Random Forest:** Reduz bastante o erro combinando múltiplas árvores em paralelo (`R² = 91.68%` no teste cru), mas ainda sofre um pouco em extremos de luxo.
* **XGBoost:** O boosting sequencial dá um salto enorme, derrubando o erro médio para `7.02%` e batendo `95.57%` de $R^2$ em apenas ~5 segundos.
* **Rede Neural MLP (Campeã):** Usando um Perceptron Multicamadas com escalonamento do alvo (`TransformedTargetRegressor` + `StandardScaler`), ela captou perfeitamente o comportamento contínuo dos preços ao longo dos anos e venceu no teste cru com **`98.77%` de $R^2$** e **`6.24%` de MAPE**!

---

## 💻 Estrutura do Repositório

```text
SI-PI6-2026-T012-CorretorIA/
├── CorretorIA_Apresentacao_PI6.ipynb # Notebook completo para o Google Colab
├── executar_sistema.bat              # Script de 2 cliques para rodar tudo no Windows
├── data/
│   ├── bairros_sp.json               # Bairros e índices base de SP
│   ├── bairros_sc.json               # Cidades/bairros e índices base de SC
│   └── dataset_sp_imoveis.csv        # Base histórica integrada (12.500 registros, 2013-2026)
├── ml/
│   ├── __init__.py
│   ├── pricing_model.py              # Motor de inferência e estimação de futuro (2027-2030)
│   ├── model_benchmark.py            # Hold-out 10% cru, GridSearchCV (5 modelos) e série anual
│   ├── benchmark_results.json        # Métricas salvas do GridSearchCV e teste cru
│   └── saved_pricing_model.joblib    # Modelo campeão salvo (Rede Neural MLP)
├── notebooks/
│   └── CorretorIA_Modelagem_e_Apresentacao.ipynb
├── scripts/
│   ├── prepare_dataset.py            # Geração e limpeza da base SP + SC
│   └── generate_notebook.py          # Gerador automático do Notebook do Google Colab
├── static/
│   ├── index.html                    # Interface web (SP/SC, estimação futura, benchmark 5 modelos)
│   ├── styles.css                    # Estilos modernos
│   └── app.js                        # Integração com a API
├── docs/
│   └── PROPOSTA_PI6.md               # Documento técnico acadêmico
├── tests/
│   └── test_api.py                   # Suíte de 6 testes automatizados
├── main.py                           # API FastAPI (v3.0.0)
├── requirements.txt                  # Dependências (inclui xgboost e matplotlib)
└── README.md                         # Este arquivo
```

---

## 🚀 Como Rodar

### Opção 1: Direto no Google Colab (Apresentação em Aula)
Clique no link abaixo para abrir direto na nuvem sem instalar nada:  
👉 **[Abrir no Google Colab](https://colab.research.google.com/github/Creolezi/SI-PI6-2026-T012-CorretorIA/blob/main/CorretorIA_Apresentacao_PI6.ipynb)**

### Opção 2: Aplicação Web Local (1 Clique no Windows)
1. Dê dois cliques em `executar_sistema.bat` na pasta do projeto.
2. Ou rode no terminal:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
   Acesse **[http://localhost:8000](http://localhost:8000)** (e documentação Swagger em `/docs`).

### Opção 3: Re-executar o Pipeline de GridSearchCV e Testes
```bash
python ml/model_benchmark.py
python tests/test_api.py
```
