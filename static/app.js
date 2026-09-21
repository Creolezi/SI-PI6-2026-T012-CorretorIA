/**
 * CorretorIA - Frontend Controller
 * Projeto Integrador 6 - SI-PI6-2026-T012
 */

document.addEventListener("DOMContentLoaded", () => {
  // Estado local da aplicação
  const state = {
    selectedUf: "SP",
    selectedTipo: "Apartamento",
    bairros: [],
    bairrosMap: {},
    margemCorretor: 6.0
  };

  // Elementos do DOM
  const bairroSelect = document.getElementById("bairro-select");
  const bairroHint = document.getElementById("bairro-hint");
  const tipoBtns = document.querySelectorAll(".type-btn");
  const areaInput = document.getElementById("area_m2");
  const areaSlider = document.getElementById("area-slider");
  const areaValLabel = document.getElementById("area-val-label");
  const anoInput = document.getElementById("ano_construcao");
  const anoSlider = document.getElementById("ano-slider");
  const idadeValLabel = document.getElementById("idade-val-label");
  const margemSlider = document.getElementById("margem-slider");
  const margemValLabel = document.getElementById("margem-val-label");
  const presetBtns = document.querySelectorAll(".preset-btn");
  const pricingForm = document.getElementById("pricing-form");
  const btnSubmit = document.getElementById("btn-submit");
  const btnText = btnSubmit.querySelector(".btn-text");
  const btnLoading = btnSubmit.querySelector(".btn-loading");

  // Elementos do Resultado
  const emptyState = document.getElementById("empty-state");
  const resultContent = document.getElementById("result-content");
  const resImovelTitulo = document.getElementById("res-imovel-titulo");
  const resImovelDetalhes = document.getElementById("res-imovel-detalhes");
  const resValorJusto = document.getElementById("res-valor-justo");
  const resPrecoM2 = document.getElementById("res-preco-m2");
  const resDiffM2 = document.getElementById("res-diff-m2");
  const resFaixaMin = document.getElementById("res-faixa-min");
  const resFaixaTarget = document.getElementById("res-faixa-target");
  const resFaixaMax = document.getElementById("res-faixa-max");
  const resTaxaComissao = document.getElementById("res-taxa-comissao");
  const resComissaoJusto = document.getElementById("res-comissao-justo");
  const resComissaoMin = document.getElementById("res-comissao-min");
  const resComissaoMax = document.getElementById("res-comissao-max");
  const resFatoresList = document.getElementById("res-fatores-list");

  // Elementos de Benchmark / Mercado
  const statTotalAmostras = document.getElementById("stat-total-amostras");
  const statMediaM2 = document.getElementById("stat-media-m2");
  const statR2Score = document.getElementById("stat-r2-score");
  const statIdadeMedia = document.getElementById("stat-idade-media");
  const marketTableBody = document.getElementById("market-table-body");

  // Formatação de Moeda Brasileira (BRL)
  const formatCurrency = (val) => {
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL"
    }).format(val);
  };

  // Inicialização
  async function init() {
    setupEventListeners();
    await loadLocations();
    await loadMarketSummary();
  }

  // Configuração dos Event Listeners
  function setupEventListeners() {
    // Sincronização Área Útil
    areaInput.addEventListener("input", (e) => {
      const val = Math.max(15, Math.min(1000, Number(e.target.value) || 15));
      areaSlider.value = Math.min(400, val);
      areaValLabel.textContent = `${val} m²`;
    });

    areaSlider.addEventListener("input", (e) => {
      areaInput.value = e.target.value;
      areaValLabel.textContent = `${e.target.value} m²`;
    });

    // Sincronização Ano de Construção
    anoInput.addEventListener("input", (e) => {
      const ano = Math.max(1950, Math.min(2026, Number(e.target.value) || 2026));
      anoSlider.value = ano;
      const idade = 2026 - ano;
      idadeValLabel.textContent = `${ano} (${idade === 0 ? "Lançamento/Novo" : idade + " anos"})`;
    });

    anoSlider.addEventListener("input", (e) => {
      const ano = Number(e.target.value);
      anoInput.value = ano;
      const idade = 2026 - ano;
      idadeValLabel.textContent = `${ano} (${idade === 0 ? "Lançamento/Novo" : idade + " anos"})`;
    });

    // Tipo de Imóvel
    tipoBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        tipoBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.selectedTipo = btn.getAttribute("data-type");
      });
    });

    // Contadores (+ / -)
    document.querySelectorAll(".counter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-target");
        const step = Number(btn.getAttribute("data-step"));
        const input = document.getElementById(targetId);
        const min = Number(input.getAttribute("min")) || 0;
        const max = Number(input.getAttribute("max")) || 10;
        let currentVal = Number(input.value) || 0;
        let newVal = Math.max(min, Math.min(max, currentVal + step));
        input.value = newVal;
      });
    });

    // Margem / Comissão do Corretor
    margemSlider.addEventListener("input", (e) => {
      const pct = Number(e.target.value);
      state.margemCorretor = pct;
      updateMargemLabel(pct);
      highlightPreset(pct);
    });

    presetBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const pct = Number(btn.getAttribute("data-margin"));
        state.margemCorretor = pct;
        margemSlider.value = pct;
        updateMargemLabel(pct);
        highlightPreset(pct);
      });
    });

    // Mudança no select de bairro para atualizar dica
    bairroSelect.addEventListener("change", () => {
      const selected = state.bairrosMap[bairroSelect.value];
      if (selected) {
        bairroHint.innerHTML = `<strong>${selected.nome} (${selected.zona}):</strong> m² base de ref. ${formatCurrency(selected.preco_m2_base)} • ${selected.descricao}`;
      }
    });

    // Submissão do Formulário
    pricingForm.addEventListener("submit", handleFormSubmit);
  }

  function updateMargemLabel(pct) {
    const isRecommended = pct === 6.0;
    margemValLabel.textContent = `${pct.toFixed(1)}% ${isRecommended ? "(CRECI Padrão)" : ""}`;
  }

  function highlightPreset(pct) {
    presetBtns.forEach((btn) => {
      if (Number(btn.getAttribute("data-margin")) === pct) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  // Carrega Localizações da API
  async function loadLocations() {
    try {
      const res = await fetch("/api/locations");
      if (!res.ok) throw new Error("Erro ao carregar localidades");
      const data = await res.json();
      
      state.bairros = data.bairros || [];
      bairroSelect.innerHTML = '<option value="" disabled selected>Selecione o bairro...</option>';

      state.bairros.forEach((b) => {
        state.bairrosMap[b.nome] = b;
        const opt = document.createElement("option");
        opt.value = b.nome;
        opt.textContent = `${b.nome} (${b.zona}) — ${formatCurrency(b.preco_m2_base)}/m²`;
        bairroSelect.appendChild(opt);
      });

      // Seleciona Moema por padrão se existir
      if (state.bairrosMap["Moema"]) {
        bairroSelect.value = "Moema";
        bairroSelect.dispatchEvent(new Event("change"));
      }
    } catch (err) {
      console.error(err);
      bairroHint.textContent = "Falha ao carregar lista de bairros do servidor.";
    }
  }

  // Carrega Resumo de Mercado da API
  async function loadMarketSummary() {
    try {
      const res = await fetch("/api/market-summary");
      if (!res.ok) return;
      const data = await res.json();

      if (data.total_registros_base) {
        statTotalAmostras.textContent = `${data.total_registros_base.toLocaleString("pt-BR")}+`;
      }
      if (data.media_geral_m2) {
        statMediaM2.textContent = formatCurrency(data.media_geral_m2);
      }
      if (data.idade_media_imoveis) {
        statIdadeMedia.textContent = `${data.idade_media_imoveis} anos`;
      }

      // Preenche tabela
      if (data.ranking_bairros && data.ranking_bairros.length > 0) {
        marketTableBody.innerHTML = "";
        data.ranking_bairros.forEach((item) => {
          const tr = document.createElement("tr");
          const isHighLiquidity = item.preco_m2_medio >= 12000;
          
          tr.innerHTML = `
            <td><strong>${item.bairro}</strong></td>
            <td><strong>${formatCurrency(item.preco_m2_medio)}</strong>/m²</td>
            <td>${formatCurrency(item.preco_medio)}</td>
            <td>${item.amostras} transações</td>
            <td>
              <span class="liquidity-pill ${isHighLiquidity ? "liquidity-high" : "liquidity-med"}">
                ${isHighLiquidity ? "Alta Demanda" : "Moderada"}
              </span>
            </td>
          `;
          marketTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.warn("Mercado não carregado:", err);
    }
  }

  // Execução da Precificação
  async function handleFormSubmit(e) {
    e.preventDefault();

    const bairro = bairroSelect.value;
    if (!bairro) {
      alert("Por favor, selecione um bairro.");
      return;
    }

    const payload = {
      estado: state.selectedUf,
      bairro: bairro,
      tipo: state.selectedTipo,
      area_m2: Number(areaInput.value),
      ano_construcao: Number(anoInput.value),
      quartos: Number(document.getElementById("quartos").value),
      banheiros: Number(document.getElementById("banheiros").value),
      vagas: Number(document.getElementById("vagas").value),
      margem_corretor_pct: state.margemCorretor
    };

    // UI Loading
    btnSubmit.disabled = true;
    btnText.style.display = "none";
    btnLoading.style.display = "inline-flex";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Erro ao calcular precificação.");
      }

      const result = await response.json();
      renderResults(result);

      // Rola suavemente até o resultado em telas menores
      if (window.innerWidth < 980) {
        document.getElementById("result-card").scrollIntoView({ behavior: "smooth" });
      }
    } catch (error) {
      alert(`Aviso: ${error.message}`);
    } finally {
      btnSubmit.disabled = false;
      btnText.style.display = "inline-flex";
      btnLoading.style.display = "none";
    }
  }

  // Renderização do Relatório Executivo
  function renderResults(res) {
    const imovel = res.imovel;
    const prec = res.precificacao;
    const comm = res.honorarios_corretor;

    // Header do Imóvel
    resImovelTitulo.textContent = `${imovel.tipo} em ${imovel.bairro}`;
    resImovelDetalhes.textContent = `${imovel.area_m2} m² • ${imovel.quartos} Dorm. • ${imovel.banheiros} Banheiros • ${imovel.vagas} Vagas • Construído em ${imovel.ano_construcao}`;

    // Valor Central
    resValorJusto.textContent = formatCurrency(prec.valor_justo);
    resPrecoM2.textContent = `${formatCurrency(prec.preco_m2)}/m²`;
    
    // Tag de Diferencial do m²
    const diff = prec.diferencial_m2_pct;
    const diffSign = diff >= 0 ? `+${diff}%` : `${diff}%`;
    resDiffM2.textContent = `${diffSign} vs. média do bairro`;
    resDiffM2.style.background = diff >= 0 ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)";
    resDiffM2.style.color = diff >= 0 ? "#065f46" : "#b91c1c";

    // Faixa de Negociação
    resFaixaMin.textContent = formatCurrency(prec.faixa_minima);
    resFaixaTarget.textContent = formatCurrency(prec.valor_justo);
    resFaixaMax.textContent = formatCurrency(prec.faixa_maxima);

    // Honorários do Corretor
    resTaxaComissao.textContent = `${comm.taxa_pct.toFixed(1)}%`;
    resComissaoJusto.textContent = formatCurrency(comm.comissao_valor_justo);
    resComissaoMin.textContent = formatCurrency(comm.comissao_faixa_min);
    resComissaoMax.textContent = formatCurrency(comm.comissao_faixa_max);

    // Fatores de Explicabilidade
    resFatoresList.innerHTML = "";
    if (res.analise_mercado && res.analise_mercado.fatores_relevantes) {
      res.analise_mercado.fatores_relevantes.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        resFatoresList.appendChild(li);
      });
    }

    // Exibir Resultado e Ocultar Empty State
    emptyState.style.display = "none";
    resultContent.style.display = "block";
  }

  // Start
  init();
});
