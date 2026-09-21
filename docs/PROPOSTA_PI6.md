# PROPOSTA DE PROJETO INTEGRADOR VI (SI-PI6-2026-T012)

**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Projeto Integrador VI (PI6) - Turma 012  
**Título do Projeto:** CorretorIA — Sistema Inteligente de Apoio à Decisão e Precificação Imobiliária com Modelos Hedônicos de Aprendizado de Máquina  
**Tema:** Aplicação de Inteligência Artificial para Otimização de Precificação e Margem Operacional de Consultores e Corretores de Imóveis  

---

## 1. RESUMO EXECUTIVO

O mercado imobiliário brasileiro caracteriza-se por acentuada assimetria de informações e forte volatilidade nos preços de venda e locação. Corretores e consultores autônomos comumente enfrentam o desafio de estimar o valor venal de um imóvel de maneira rápida, fundamentada e justa — equilibrando as expectativas muitas vezes superestimadas do proprietário com a liquidez exigida pelo comprador. 

O **CorretorIA** é um sistema preditivo desenvolvido sob medida para o profissional imobiliário. A partir da modelagem hedônica (regressão multivariada com algoritmos de *Machine Learning*), o sistema ingere atributos estruturais (metragem quadrada, idade do imóvel/ano de construção, cômodos, vagas de garagem e tipologia) e atributos espaciais (bairro e zoneamento urbano) para estimar:
1. O **Preço Central Justo de Mercado** (R$ e R$/m²);
2. A **Faixa de Negociação Recomendada** (preço mínimo de liquidez acelerada vs. teto sugerido);
3. A **Projeção de Honorários e Margem** do corretor (comissão em R$ nos diferentes cenários);
4. O **Parecer Técnico Justificado** (fatores de explicabilidade para apresentação ao cliente proprietário).

---

## 2. DELIMITAÇÃO DO ESCOPO & RESPOSTA À DEMANDA MULTI-ESTADO

### 2.1 Demanda Inicial de Cobertura Nacional
Durante o alinhamento da proposta, foi levantada a viabilidade de estender a cobertura do sistema para múltiplos estados ou para todo o território nacional.

### 2.2 Estratégia Técnica da 1ª Entrega (MVP Homologado em SP)
Para garantir rigor estatístico, confiabilidade de dados e entrega de um produto funcional (*Working Software*), optou-se fundamentadamente por:
- **Base Piloto (São Paulo - SP):** São Paulo representa a maior densidade de transações imobiliárias e volume de dados abertos da América Latina, constituindo o cenário ideal para calibração dos hiperparâmetros e validação do modelo.
- **Arquitetura Modular Multi-Estado (Desde a Sprint 1):** O esquema de dados e as rotas de API foram desenhados com desacoplamento regional (`estado`, `cidade`, `bairro`), de modo que a inclusão de Minas Gerais (Belo Horizonte), Santa Catarina (Florianópolis / Balneário Camboriú) e outros estados nas etapas seguintes exija apenas a anexação dos novos datasets ao pipeline de treino, sem qualquer refatoração estrutural da aplicação.

---

## 3. REQUISITOS DO SISTEMA

### 3.1 Requisitos Funcionais (RF)
- **[RF01] Inserção Parametrizada de Atributos:** O consultor pode definir bairro, tipo (Apartamento, Casa, Cobertura), metragem ($m^2$), ano de construção, dormitórios, banheiros e vagas de garagem.
- **[RF02] Cálculo de Valuation com IA:** O sistema processa os atributos através do modelo de regressão treinado e devolve a estimativa de preço em milissegundos.
- **[RF03] Faixa Dinâmica de Negociação:** Além do preço central, o sistema calcula faixas de tolerância (mínima de liquidez e máxima negociável) baseadas no erro residual estatístico ($MAPE$).
- **[RF04] Calculadora de Honorários do Corretor:** O usuário pode simular percentuais de comissão (ex: 5%, 6% tabela CRECI, até 10%) e visualizar a rentabilidade em cada cenário de negociação.
- **[RF05] Justificativa e Explicabilidade:** Geração de tópicos explicativos apontando o impacto positivo ou depreciativo das características do imóvel.
- **[RF06] Painel de Benchmarks de Mercado:** Exibição do panorama com m² médio e liquidez dos bairros paulistanos.
- **[RF07] Emissão de Parecer Técnico:** Suporte à impressão e exportação formatada para apresentação ao cliente final.

### 3.2 Requisitos Não Funcionais (RNF)
- **[RNF01] Desempenho:** Tempo de resposta da API inferior a 250ms por predição.
- **[RNF02] Precisão Estatística:** Coeficiente de determinação $R^2 \ge 0.90$ e Erro Médio Percentual ($MAPE \le 12\%$).
- **[RNF03] Interface Responsiva:** Layout adaptável para notebooks, tablets e smartphones.
- **[RNF04] Arquitetura Desacoplada:** Backend construído com API RESTful em FastAPI e frontend SPA desacoplado.

---

## 4. METODOLOGIA E TECNOLOGIAS

- **Linguagem & Backend:** Python 3.12 + FastAPI (alta performance assíncrona, documentação OpenAPI/Swagger nativa).
- **Inteligência Artificial / Estatística:** Scikit-Learn, Pandas, NumPy, Joblib.
  - *Pipeline:* `ColumnTransformer` (One-Hot Encoding para atributos categóricos e StandardScaler para métricas contínuas) acoplado ao regressor `RandomForestRegressor`.
- **Frontend:** HTML5 semântico, CSS3 com variáveis dinâmicas e design moderno, JavaScript ES6 puro (sem sobrecarga de dependências pesadas).
- **Servidor:** Uvicorn ASGI.

---

## 5. RESULTADOS OBTIDOS NA 1ª ENTREGA (SPRINT 1)

1. **Pipeline de Dados Operacional:** Criação e estruturação da base inicial com 7.500 registros imobiliários de São Paulo abrangendo a série histórica de 2013 a 2026.
2. **Modelo Preditivo Treinado e Homologado:**
   - Coeficiente de Determinação: **$R^2 = 0.9758$ (97.58% da variância explicada)**;
   - Erro Médio Absoluto Percentual: **$MAPE = 9.63\%$**;
   - Persistência serializada com `joblib` para inicialização imediata em produção.
3. **Plataforma Web Interativa:** Sistema completo em execução, permitindo simulação em tempo real de imóveis em Moema, Pinheiros, Itaim Bibi, Vila Mariana, Tatuapé, Perdizes e demais regiões nobres e residenciais de SP.

---

## 6. CRONOGRAMA DE ENTREGAS (SPRINTS)

| Sprint | Período | Entregáveis | Status |
| :--- | :--- | :--- | :--- |
| **Sprint 1 (Atual)** | Semanas 1 a 3 | Formulação do projeto, pipeline de dados, modelo preditivo calibrado de SP, API FastAPI e Frontend funcional do Corretor. | **Concluído** |
| **Sprint 2** | Semanas 4 a 6 | Ingestão do dataset completo bruto (ITBI/Geosampa), enriquecimento com métricas de condomínio/IPTU e testes de carga. | Planejado |
| **Sprint 3** | Semanas 7 a 9 | Expansão piloto para segundo estado (Minas Gerais ou Santa Catarina), comparativo regional e geração de PDF assinado. | Planejado |
| **Sprint 4** | Semanas 10 a 12 | Validação com corretores parceiros, testes de usabilidade, relatório técnico final e preparação para a banca. | Planejado |
