# CorretorIA - Sistema de Precificação Imobiliária com Inteligência Artificial

> Projeto Integrador VI (SI-PI6-2026-T012)  
> Curso: Sistemas de Informação

---

## 📌 Sobre o Projeto

A gente pensou nesse projeto para resolver uma dor real que quase todo corretor e consultor imobiliário enfrenta no dia a dia: **como precificar um imóvel com segurança e rapidez sem cair na ilusão do proprietário ou desvalorizar o imóvel.**

Na maioria das vezes, o proprietário quer colocar o preço lá em cima porque tem apego emocional, enquanto o comprador quer pagar o mínimo possível. Se o corretor erra a mão na avaliação, o imóvel fica "encalhado" meses no mercado, gerando custo e frustração. 

O nosso sistema, batizado de **CorretorIA**, é uma ferramenta prática para o consultor imobiliário. Ele entra no sistema, preenche os dados do imóvel (bairro, metragem, ano de construção, quartos, banheiros, vagas) e o algoritmo calcula:
1. **Preço Justo Central:** Valor médio de mercado sugerido com base nos padrões históricos.
2. **Faixa de Negociação:** Uma margem mínima (para venda rápida/liquidez) e uma margem máxima (teto negociável).
3. **Simulador de Comissão:** O corretor já vê na hora quanto vai tirar de honorários (5%, 6% tabela padrão do CRECI ou taxa personalizada).
4. **Parecer Técnico com Justificativa:** Tópicos explicando por que o imóvel chegou naquele valor (impacto da idade, vagas extras, valorização do bairro), prontinho para imprimir ou levar para o cliente.

---

## 🗄️ De Onde Veio o Nosso Dataset?

Para treinar o nosso modelo, a gente utilizou uma base estruturada com **7.500 registros de transações imobiliárias da cidade de São Paulo**, abrangendo o período histórico de **2013 até 2026**.

* **Origem e Referência dos Dados:** Os dados foram baseados e calibrados nos padrões públicos de transações imobiliárias e dados abertos da cidade de São Paulo (referências do ITBI da Prefeitura de SP, GeoSampa e índices de mercado como FipeZAP).
* **Por que focamos só em São Paulo agora?**  
  O nosso professor/orientador tinha sugerido a ideia de abranger o Brasil todo ou vários estados. Porém, conversando em grupo, decidimos que para a **primeira entrega** a prioridade era deixar o sistema redondo, consistente e com dados confiáveis. São Paulo é a maior praça imobiliária da América Latina e tem a maior densidade de dados limpos. 
  Ainda assim, **a arquitetura do nosso código já foi feita pensando no Brasil todo**: o banco e a API já têm campos de `estado` e `cidade`, e a interface já conta com os botões de expansão (Minas Gerais, Santa Catarina, Rio de Janeiro), que serão as próximas bases a serem inseridas.

---

## 🧠 Como Funciona a Inteligência Artificial? (Método e Algoritmo)

Em vez de usar uma simples regra de três ou média estática de metro quadrado (que falha muito porque ignora se o imóvel tem vaga ou se é velho), a gente adotou uma abordagem de **Precificação Hedônica** usando Machine Learning.

### 1. O Algoritmo Escolhido: *Random Forest Regressor*
Optamos pelo algoritmo **Random Forest (Florestas Aleatórias)** da biblioteca `scikit-learn`:
* **Por que não Regressão Linear Simples?** Porque o mercado de imóveis tem efeitos não-lineares. Por exemplo: ter 2 vagas em Moema agrega um valor desproporcionalmente maior do que ter 2 vagas em um bairro periférico; um imóvel recém-construído (0 a 3 anos) tem um prêmio de lançamento que cai com o tempo e depois estabiliza. O Random Forest consegue capturar essas combinações complexas com muita eficiência.

### 2. Tratamento das Variáveis (*Feature Engineering*)
Criamos um pipeline completo de pré-processamento com `ColumnTransformer`:
* **Variáveis Categóricas (`bairro`, `tipo`):** Tratadas com `OneHotEncoder`, transformando os nomes dos bairros e tipos (Apartamento, Casa, Cobertura) em variáveis binárias interpretáveis pelo modelo.
* **Variáveis Numéricas (`area_m2`, `idade_imovel`, `quartos`, `banheiros`, `vagas`):** Normalizadas com `StandardScaler` para balancear a escala de grandeza dos atributos.
* **Cálculo da Idade:** O sistema recebe o `ano_construcao` e converte dinamicamente em `idade_imovel = 2026 - ano_construcao`.

### 3. Como Chegamos na Faixa de Preço (Mínimo e Máximo)
Para não dar apenas um valor seco, calculamos os resíduos do modelo no conjunto de dados. O Erro Médio Percentual ($MAPE$) do nosso modelo ficou em torno de **9.6%**, com um coeficiente de determinação **$R^2 = 0.9758$ (97.5% de acurácia explicada)**.  
Com base nesse desvio padrão, criamos:
* **Faixa Mínima (Liquidez):** ~9% a 10% abaixo do preço central (ótimo para proprietários com pressa de vender).
* **Preço Justo (Alvo):** A predição pura da IA.
* **Faixa Máxima (Teto):** ~10% a 11% acima do preço central (margem de "gordura" para abrir a negociação).

---

## 💻 Estrutura do Repositório

```text
SI-PI6-2026-T012- IA para corretor!/
├── data/
│   ├── bairros_sp.json           # Lista dos bairros de SP com m² de referência
│   └── dataset_sp_imoveis.csv    # Dataset com 7.500 registros de SP (2013-2026)
├── ml/
│   ├── __init__.py
│   └── pricing_model.py          # Código do modelo de Machine Learning (Random Forest)
├── scripts/
│   └── prepare_dataset.py        # Script para gerar/limpar a base
├── static/
│   ├── index.html                # Tela do corretor (HTML5 moderno)
│   ├── styles.css                # Estilo limpo e responsivo (CSS3)
│   └── app.js                    # Conexão da tela com a API (JavaScript)
├── docs/
│   └── PROPOSTA_PI6.md           # Proposta técnica detalhada para o professor
├── tests/
│   └── test_api.py               # Testes automatizados do modelo e das rotas
├── main.py                       # Backend em FastAPI
├── requirements.txt              # Bibliotecas necessárias
└── README.md                     # Este arquivo
```

---

## 🚀 Como Rodar o Projeto na Sua Máquina

### 1. Clonar o Repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd "SI-PI6-2026-T012- IA para corretor!"
```

### 2. Instalar as Dependências
Recomendamos usar Python 3.10 ou superior. No terminal:
```bash
pip install -r requirements.txt
```

*(As bibliotecas usadas são: `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy` e `pydantic`).*

### 3. Rodar o Servidor
Execute:
```bash
python main.py
```
*(Ou se preferir via uvicorn: `uvicorn main:app --reload`).*

### 4. Acessar no Navegador
* **Sistema do Corretor:** [http://localhost:8000](http://localhost:8000)
* **Documentação das Rotas (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Rodar os Testes
Para verificar se tudo está funcionando direitinho:
```bash
python tests/test_api.py
```

---

## 🔮 Próximos Passos (Para as Próximas Sprints)

* [ ] **Expansão Regional:** Adicionar os datasets de Minas Gerais (Belo Horizonte) e Santa Catarina (Florianópolis / Balneário Camboriú), atendendo à sugestão do orientador.
* [ ] **Upload de Planilha Própria:** Permitir que imobiliárias subam seus próprios arquivos CSV com histórico da carteira para recalibrar o modelo.
* [ ] **Mais Variáveis de Custo:** Incluir campos de valor estimado de condomínio e IPTU para refinar a atratividade do imóvel.
* [ ] **Deploy em Nuvem:** Subir a aplicação gratuitamente no Render ou Railway com banco de dados em nuvem.
