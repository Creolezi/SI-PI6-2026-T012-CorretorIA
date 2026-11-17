# CorretorIA - Sistema de Precificação Imobiliária com Inteligência Artificial

> Projeto Integrador VI (SI-PI6-2026-T012)  
> Curso: Sistemas de Informação • Cobertura Nacional Piloto: 6 Estados (`SP`, `SC`, `MG`, `RJ`, `DF`, `PR`) & Estimação Futura (`2026–2035`)

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Creolezi/SI-PI6-2026-T012-CorretorIA/blob/main/CorretorIA_Apresentacao_PI6.ipynb)

---

## 📌 Sobre o Projeto

A gente pensou nesse projeto para resolver uma dor real que quase todo corretor e consultor imobiliário enfrenta no dia a dia: **como precificar um imóvel com segurança e rapidez sem cair na ilusão do proprietário ou desvalorizar o imóvel.**

O nosso sistema, batizado de **CorretorIA**, ajuda o consultor imobiliário a avaliar imóveis em **6 estados brasileiros** (`SP`, `SC`, `MG`, `RJ`, `DF` e `PR`) com embasamento estatístico, entregando:
1. **Preço Justo Central:** Valor de mercado estimado pelo modelo campeão para 2026 ou para o ano futuro selecionado.
2. **Faixa de Negociação:** Margem mínima (liquidez rápida) e teto negociável.
3. **Seletor de Ano da Estimação Futura (`2026–2035`):** Tanto na interface web quanto nos sliders do Google Colab, o corretor pode escolher qualquer ano entre **2026 e 2035** e ver a projeção de valorização patrimonial daquele imóvel.
4. **Simulador de Comissão:** Cálculo direto dos honorários do consultor (6% padrão CRECI ou personalizado).
5. **Parecer Técnico Explicativo:** Fatores que mais influenciaram o valor do imóvel.

---

## 🗄️ De Onde Vieram os Nossos Datasets? (6 Estados — 18.000 Registros)

Unificamos uma base histórica com **18.000 registros de transações (2013 até 2026)** distribuídos em 6 estados:
* **São Paulo (`SP` - 5.000 registros):** Capital paulista (Moema, Itaim Bibi, Pinheiros, Jardins, Vila Mariana, Tatuapé, etc.), calibrada nos padrões do ITBI da Prefeitura de SP e GeoSampa.
* **Santa Catarina (`SC` - 3.500 registros):** Balneário Camboriú (Barra Sul, Centro, Pioneiros), Itapema (Meia Praia, Centro), Florianópolis (Jurerê Internacional, Campeche, Agronômica, Centro, Itacorubi), Joinville e Blumenau.
* **Rio de Janeiro (`RJ` - 3.000 registros):** Leblon, Ipanema, Lagoa, Botafogo, Copacabana, Barra da Tijuca, Tijuca e Niterói (Icaraí).
* **Minas Gerais (`MG` - 2.500 registros):** Belo Horizonte (Belvedere, Lourdes, Savassi, Santo Agostinho, Funcionários, Buritis, Pampulha) e Uberlândia (Granja Marileusa).
* **Paraná (`PR` - 2.000 registros):** Curitiba (Batel, Ecoville, Bigorrilho, Cabral, Água Verde, Centro Cívico) e Londrina (Gleba Palhano).
* **Distrito Federal (`DF` - 2.000 registros):** Brasília (Noroeste, Sudoeste, Lago Sul, Asa Sul, Asa Norte, Águas Claras).

---

## 📈 Variação Anual (2013–2026) & Estimação de Futuro (2027–2035)

Para atender às discussões da aula:
1. **Plot separado por ano (`ano_transacao`):** Separamos as transações de **2013 até 2026** em um grid por estado para visualizar claramente como o preço médio e a variação do m² evoluíram ano a ano.
2. **Estimação de Futuro (`2027–2035`):** Ajustamos uma tendência temporal sobre a série histórica anual de cada estado (com banda de intervalo de confiança) e disponibilizamos um seletor interativo (`2026–2035`) no simulador para projetar o valor futuro do imóvel.

---

## 🧪 Ambiente de Teste Isolado (Hold-out 10% Cru) & `GridSearchCV`

1. **Separação de 10% do Dataset "Cru" (`Hold-out`):**
   * Antes de qualquer ajuste ou busca de hiperparâmetros, **tiramos 10% do dataset bruto (`1.800` imóveis)** de forma estratificada pelos 6 estados e deixamos esse bloco totalmente de lado.
   * Assim, os modelos avaliados nunca tiveram contato com esses dados, garantindo avaliação cega sem viés.
2. **Busca de Hiperparâmetros com `GridSearchCV` (5-Fold nos 90% restantes):**
   * Usamos os **90% restantes (`16.200` imóveis)** dentro do `GridSearchCV(cv=5)` para variar os hiperparâmetros em grade de todos os modelos e escolher automaticamente a melhor configuração de cada um.

---

## 🤖 Comparativo dos 5 Modelos (incluindo `XGBoost` e `Rede Neural MLP`)

Todas as métricas abaixo foram calculadas para cada modelo otimizado pelo `GridSearchCV` e avaliadas no **conjunto de teste cru de 10% (`1.800` imóveis inéditos)**:

| Modelo de IA | Melhores Parâmetros (`GridSearchCV`) | R² CV (90%) | R² Teste Cru (10%) | MAE (Teste Cru) | MAPE (Teste Cru) | RMSE (Teste Cru) | Tempo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rede Neural MLP** 🏆 | `{'alpha': 0.001, 'hidden_layer_sizes': [128, 64], 'learning_rate_init': 0.005}` | **98.71%** | **97.88%** | **R$ 75.344,86** | **6.73%** | **R$ 167.369,57** | 30.82s |
| **XGBoost** | `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150}` | 96.41% | 94.93% | R$ 100.465,32 | 7.78% | R$ 258.748,19 | 8.15s |
| **Random Forest** | `{'max_depth': 16, 'min_samples_split': 4, 'n_estimators': 120}` | 91.84% | 87.26% | R$ 179.760,35 | 14.82% | R$ 410.108,37 | 80.60s |
| **Regressão Linear (Ridge)** | `{'alpha': 1.0}` | 89.20% | 85.16% | R$ 203.592,63 | 24.70% | R$ 442.602,36 | 4.44s |
| **Árvore de Decisão** | `{'max_depth': 16, 'min_samples_split': 10}` | 86.61% | 82.77% | R$ 223.602,62 | 17.53% | R$ 476.873,63 | 1.45s |

### O que a gente concluiu comparando os 5 modelos:
* **Árvore de Decisão & Regressão Linear (Ridge):** São rápidas, mas erram mais quando precisam combinar 6 escalas estaduais diferentes, bairro, metragem e ano (`MAPE` de 17.53% e 24.70%).
* **Random Forest:** Reduz bastante o erro combinando múltiplas árvores em paralelo (`R² = 87.26%` no teste cru), mas ainda sofre nos extremos de luxo entre estados.
* **XGBoost:** O boosting sequencial dá um salto expressivo, derrubando o erro médio para `7.78%` e atingindo `94.93%` de $R^2$ em ~8 segundos.
* **Rede Neural MLP (Campeã):** Com a rede `[128, 64]` encontrada pelo `GridSearchCV` e escalonamento do alvo (`TransformedTargetRegressor` + `StandardScaler`), captou perfeitamente o comportamento contínuo dos preços nos 6 estados e venceu no teste cru com **`97.88%` de $R^2$** e **`6.73%` de MAPE**!

---

## 💻 Estrutura do Repositório

```text
SI-PI6-2026-T012-CorretorIA/
├── CorretorIA_Apresentacao_PI6.ipynb # Notebook completo para o Google Colab (6 estados + 2026-2035)
├── executar_sistema.bat              # Script de 2 cliques para rodar tudo no Windows
├── data/
│   ├── bairros_sp.json               # Bairros e índices base de SP
│   ├── bairros_sc.json               # Cidades/bairros de SC
│   ├── bairros_mg.json               # Cidades/bairros de MG
│   ├── bairros_rj.json               # Cidades/bairros do RJ
│   ├── bairros_df.json               # Regiões/bairros do DF
│   ├── bairros_pr.json               # Cidades/bairros do PR
│   └── dataset_sp_imoveis.csv        # Base histórica integrada (18.000 registros, 2013-2026)
├── ml/
│   ├── __init__.py
│   ├── pricing_model.py              # Motor de inferência (6 estados) e estimação futura (2026-2035)
│   ├── model_benchmark.py            # Hold-out 10% cru, GridSearchCV (5 modelos) e série anual
│   ├── benchmark_results.json        # Métricas salvas do GridSearchCV e teste cru
│   └── saved_pricing_model.joblib    # Modelo campeão salvo (Rede Neural MLP)
├── notebooks/
│   └── CorretorIA_Modelagem_e_Apresentacao.ipynb
├── scripts/
│   ├── prepare_dataset.py            # Geração e limpeza da base dos 6 estados
│   └── generate_notebook.py          # Gerador automático do Notebook do Google Colab
├── static/
│   ├── index.html                    # Interface web (6 estados, seletor 2026-2035, benchmark 5 modelos)
│   ├── styles.css                    # Estilos modernos
│   └── app.js                        # Integração com a API
├── docs/
│   └── PROPOSTA_PI6.md               # Documento técnico acadêmico
├── tests/
│   └── test_api.py                   # Suíte de 6 testes automatizados (6 estados e projeções)
├── main.py                           # API FastAPI (v3.1.0)
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
