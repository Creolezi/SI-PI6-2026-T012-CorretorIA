# CorretorIA - Sistema de Precificação Imobiliária com Inteligência Artificial

> Projeto Integrador VI (SI-PI6-2026-T012)  
> Curso: Sistemas de Informação • Sprint 2 (SP & SC + Validação Cruzada)

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Creolezi/SI-PI6-2026-T012-CorretorIA/blob/main/CorretorIA_Apresentacao_PI6.ipynb)

---

## 📌 Sobre o Projeto

A gente pensou nesse projeto para resolver uma dor real que quase todo corretor e consultor imobiliário enfrenta no dia a dia: **como precificar um imóvel com segurança e rapidez sem cair na ilusão do proprietário ou desvalorizar o imóvel.**

Na maioria das vezes, o proprietário quer colocar o preço lá em cima porque tem apego emocional, enquanto o comprador quer pagar o mínimo possível. Se o corretor erra a mão na avaliação, o imóvel fica "encalhado" meses no mercado, gerando custo e frustração. 

O nosso sistema, batizado de **CorretorIA**, é uma ferramenta prática para o consultor imobiliário. Ele entra no sistema, seleciona o estado (São Paulo ou Santa Catarina), preenche os dados do imóvel (bairro, metragem, ano de construção, quartos, banheiros, vagas) e o algoritmo calcula:
1. **Preço Justo Central:** Valor médio de mercado sugerido com base nos padrões históricos.
2. **Faixa de Negociação:** Uma margem mínima (para venda rápida/liquidez) e uma margem máxima (teto negociável).
3. **Simulador de Comissão:** O corretor já vê na hora quanto vai tirar de honorários (5%, 6% tabela padrão do CRECI ou taxa personalizada).
4. **Parecer Técnico com Justificativa:** Tópicos explicando por que o imóvel chegou naquele valor (impacto da idade, vagas extras, valorização do bairro), prontinho para imprimir ou levar para o cliente.

---

## 🗄️ De Onde Vieram os Nossos Datasets? (SP & SC)

Na Sprint 1, a gente validou a base piloto de São Paulo. Agora na **Sprint 2**, atendendo à sugestão do orientador, expandimos a solução incluindo **Santa Catarina (SC)**, unificando uma base histórica com **12.500 registros de transações (2013 até 2026)**:

* **São Paulo (SP - 7.500 registros):** Calibrados nas declarações de ITBI da Prefeitura de SP, GeoSampa e índices de mercado como FipeZAP.
* **Santa Catarina (SC - 5.000 registros):** Cidades de destaque nacional em valorização imobiliária:
  * **Balneário Camboriú:** Maior valor de metro quadrado do Brasil (Barra Sul, Centro, Pioneiros).
  * **Itapema:** Segundo metro quadrado mais valorizado de SC (Meia Praia e Centro).
  * **Florianópolis:** Capital catarinense (Jurerê Internacional, Campeche, Agronômica, Centro, Itacorubi).
  * **Joinville e Blumenau:** Polos industriais, universitários e comerciais de alta densidade no estado.

---

## 🔬 Validação Cruzada (5-Fold Cross-Validation) & Comparativo Didático de 3 Modelos

Para garantir rigor acadêmico e provar que o nosso modelo não estava com resultados enviesados por "sorte" em uma divisão simples de 80/20, adotamos **Validação Cruzada com 5 Dobras (5-Fold Cross-Validation)**.

### 📚 O que é e por que usamos a Validação Cruzada (K-Fold 5)?
Em vez de treinar uma única vez, a base de 12.500 registros é fatiada em 5 blocos iguais (folds) de 2.500 amostras cada:
1. O algoritmo treina em 4 blocos (10.000 amostras) e é testado no bloco restante (2.500 amostras).
2. O processo se repete 5 vezes, alternando o bloco de teste a cada rodada.
3. As métricas finais são a média e o desvio padrão obtidos nas 5 rodadas ($\mu \pm \sigma$), provando a consistência real do aprendizado.

---

### 🤖 Os 3 Modelos Avaliados e Explicados Didaticamente

Avaliamos três abordagens pedagógicas sobre os mesmos dados:

| Modelo de Inteligência Artificial | Tipo / Família | R² Médio (%) | Desvio R² (±%) | Erro Médio (MAE em R$) | Erro Percentual (MAPE) | Tempo de Treino |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Ensemble)** 🏆 | Ensemble (Bagging) | **89.11%** | **± 1.63%** | **R$ 197.551,32** | **16.96%** | **0.78s** |
| **Regressão Linear (Ridge)** | Linear Regularizado | 86.33% | ± 1.01% | R$ 242.727,10 | 27.14% | 0.04s |
| **Árvore de Decisão Simples** | Árvore de Decisão | 81.86% | ± 2.48% | R$ 243.911,24 | 20.60% | 0.06s |

#### 1. Regressão Linear / Ridge (Baseline Clássico)
* **Como funciona:** Modela o preço como uma soma ponderada direta dos atributos ($y = w_1 x_1 + \dots + w_n x_n + b$).
* **Por que usamos:** Serve como o ponto de partida mínimo (baseline) da ciência de dados.
* **O que aprendemos:** É ultrarrápida para treinar (0.04s), mas tem dificuldade com comportamentos não-lineares. Por exemplo, ela assume que o impacto de uma vaga de garagem é constante, ignorando que ter 2 vagas na Barra Sul (Balneário Camboriú) valoriza muito mais do que ter 2 vagas em um bairro periférico. Por isso, seu erro percentual ($MAPE$) foi o mais alto (27.14%).

#### 2. Árvore de Decisão Simples (Decision Tree)
* **Como funciona:** Divide os dados em regras condicionais (*"se bairro = Barra Sul e área > 100..."*).
* **Por que usamos:** Para testar um modelo não-linear básico baseado em regras de negócio.
* **O que aprendemos:** Consegue captar nichos e regras não-lineares, derrubando o erro percentual para 20.60%. Porém, por ser uma árvore solitária, ela oscila bastante dependendo da amostra (teve o maior desvio padrão entre as dobras: $\pm 2.48\%$), mostrando risco de overfitting.

#### 3. Random Forest (Ensemble de Bagging — O Campeão)
* **Como funciona:** Combina 100 árvores de decisão diferentes. Cada árvore é treinada com uma amostra aleatória dos dados e um subconjunto aleatório de variáveis. A predição final é a média de todas as árvores.
* **Por que venceu:** A média de múltiplas árvores cancela os ruídos individuais. Ele alcançou o maior $R^2$ (**89.11%**), o menor erro médio em reais (**R$ 197.551**) e o menor erro percentual (**16.96%**), sendo escolhido como o modelo oficial para o sistema em produção.

---

## 💻 Formatos de Apresentação para o Professor

O projeto está estruturado em **duas frentes** para a apresentação da banca/professor:

1. 📓 **Jupyter Notebook Acadêmico (`CorretorIA_Apresentacao_PI6.ipynb`):**
   * Perfeito para mostrar o passo a passo da IA célula por célula ao professor:
     * Carga dos dados unificados (12.500 registros de SP e SC).
     * Análise Exploratória (ranking dos maiores m² de Balneário Camboriú, Itapema, Floripa e SP).
     * Pré-processamento com `ColumnTransformer` (`OneHotEncoder` + `StandardScaler`).
     * Execução da **Validação Cruzada (5-Fold Cross-Validation)** nos 3 modelos.
     * Tabela comparativa e conclusões didáticas.
     * Simulador interativo em Python testando imóveis em Moema (SP) e Balneário Camboriú (SC).

2. 🌐 **Aplicação Web Interativa (FastAPI + Frontend Moderno):**
   * Dashboard com alternância em 1 clique entre **São Paulo (SP)** e **Santa Catarina (SC)**.
   * Seção visual de **Benchmark Científico** com tabela e cards explicativos dos modelos.
   * Executável de 1 clique: **`executar_sistema.bat`**.

---

## 💻 Estrutura do Repositório

```text
SI-PI6-2026-T012-CorretorIA/
├── CorretorIA_Apresentacao_PI6.ipynb # Jupyter Notebook para apresentar ao professor
├── executar_sistema.bat              # Script de 2 cliques para rodar tudo no laptop
├── data/
│   ├── bairros_sp.json               # Lista dos bairros de SP com m² de referência
│   ├── bairros_sc.json               # Lista das cidades/bairros de SC (BC, Floripa, Itapema, etc.)
│   └── dataset_sp_imoveis.csv        # Dataset unificado com 12.500 registros (2013-2026)
├── ml/
│   ├── __init__.py
│   ├── pricing_model.py              # Motor de inferência multi-estado (SP e SC)
│   ├── model_benchmark.py            # Pipeline de validação cruzada K-Fold 5 e treino dos 3 modelos
│   ├── benchmark_results.json        # Relatório de validação cruzada serializado
│   └── saved_pricing_model.joblib    # Modelo campeão (Random Forest) serializado
├── notebooks/
│   └── CorretorIA_Modelagem_e_Apresentacao.ipynb
├── scripts/
│   ├── prepare_dataset.py            # Geração da base unificada SP + SC
│   └── generate_notebook.py          # Script gerador do Jupyter Notebook
├── static/
│   ├── index.html                    # Interface web com seletor SP/SC e aba de benchmark
│   ├── styles.css                    # Estilos modernos e responsivos
│   └── app.js                        # Controlador JavaScript da aplicação
├── docs/
│   └── PROPOSTA_PI6.md               # Documento técnico da proposta para a banca
├── tests/
│   └── test_api.py                   # Suíte de 5 testes automatizados (SP, SC, Benchmark)
├── main.py                           # Backend em FastAPI
├── requirements.txt                  # Bibliotecas necessárias
└── README.md                         # Este arquivo
```

---

## 🚀 Como Rodar o Sistema

### Opção A: Execução em 1 Clique (Aplicação Web)
Dê **dois cliques no arquivo `executar_sistema.bat`** na pasta do projeto.  
O navegador abrirá automaticamente em [http://localhost:8000](http://localhost:8000).

### Opção B: Rodar o Jupyter Notebook da Apresentação
Abra o arquivo **`CorretorIA_Apresentacao_PI6.ipynb`** no VS Code ou via terminal:
```bash
jupyter notebook CorretorIA_Apresentacao_PI6.ipynb
```

---

## 🔮 Próximos Passos (Sprints Futuras)

* [x] **Sprint 1:** Base piloto de São Paulo (SP) e MVP funcional da precificação.
* [x] **Sprint 2:** Expansão para Santa Catarina (SC), pipeline de validação cruzada 5-Fold e comparativo didático de 3 modelos de IA.
* [ ] **Sprint 3:** Upload de planilha própria do corretor para recalibração local.
* [ ] **Sprint 4:** Deploy público em nuvem (Render/Railway).
