/**
 * CorretorIA - Frontend Controller
 * Projeto Integrador 6 - SI-PI6-2026-T012 - Sprint 2 (SP & SC)
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
  const statePills = document.querySelectorAll(".state-pill:not(.disabled)");
  const lblEstadoAtivo = document.getElementById("lbl-estado-ativo");
  const labelBairroTitulo = document.getElementById("label-bairro-titulo");
  const marketTitleText = document.getElementById("market-title-text");
  const btnOpenBenchmark = document.getElementById("btn-open-benchmark");
  const benchmarkSection = document.getElementById("benchmark-section");
  const benchmarkTableBody = document.getElementById("benchmark-table-body");

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
    }).format(val || 0);
  };

  // Inicialização
  async function init() {
    setupEventListeners();
    await loadLocations();
    await loadMarketSummary();
    await loadBenchmarkResults();
  }

  function setupEventListeners() {
    // Alternância de Estado (SP / SC)
    statePills.forEach((pill) => {
      pill.addEventListener("click", async () => {
        const uf = pill.getAttribute("data-uf");
        if (uf === state.selectedUf) return;

        statePills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        state.selectedUf = uf;

        // Atualiza labels de contexto
        if (uf === "SC") {
          lblEstadoAtivo.textContent = "Santa Catarina (SC)";
          labelBairroTitulo.textContent = "Bairro / Cidade (Santa Catarina)";
          marketTitleText.textContent = "Panorama Imobiliário — Santa Catarina (FipeZAP)";
        } else {
          lblEstadoAtivo.textContent = "São Paulo (SP)";
          labelBairroTitulo.textContent = "Bairro (São Paulo)";
          marketTitleText.textContent = "Panorama Imobiliário — São Paulo (Capital)";
        }

        await loadLocations();
        await loadMarketSummary();
      });
    });

    // Botão de atalho para a seção de benchmark
    if (btnOpenBenchmark) {
      btnOpenBenchmark.addEventListener("click", () => {
        benchmarkSection.scrollIntoView({ behavior: "smooth" });
      });
    }

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

    // Mudança no select de bairro
    bairroSelect.addEventListener("change", () => {
      const selected = state.bairrosMap[bairroSelect.value];
      if (selected) {
        const prefixoCidade = selected.cidade && selected.cidade !== "São Paulo" ? `${selected.cidade} • ` : "";
        bairroHint.innerHTML = `<strong>${prefixoCidade}${selected.nome} (${selected.zona}):</strong> m² base de ref. ${formatCurrency(selected.preco_m2_base)} • ${selected.descricao}`;
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

  // Carrega Localizações da API filtrando pelo estado selecionado
  async function loadLocations() {
    try {
      const res = await fetch(`/api/locations?estado=${state.selectedUf}`);
      if (!res.ok) throw new Error("Erro ao carregar localidades");
      const data = await res.json();
      
      state.bairros = data.bairros || [];
      state.bairrosMap = {};
      bairroSelect.innerHTML = '<option value="" disabled selected>Selecione o bairro...</option>';

      state.bairros.forEach((b) => {
        state.bairrosMap[b.nome] = b;
        const opt = document.createElement("option");
        opt.value = b.nome;
        const cidadeTxt = b.cidade && b.cidade !== "São Paulo" ? `[${b.cidade}] ` : "";
        opt.textContent = `${cidadeTxt}${b.nome} (${b.zona}) — ${formatCurrency(b.preco_m2_base)}/m²`;
        bairroSelect.appendChild(opt);
      });

      // Seleciona um bairro padrão por estado
      if (state.selectedUf === "SC") {
        if (state.bairrosMap["Barra Sul"]) {
          bairroSelect.value = "Barra Sul";
        } else if (state.bairrosMap["Jurerê Internacional"]) {
          bairroSelect.value = "Jurerê Internacional";
        }
      } else {
        if (state.bairrosMap["Moema"]) {
          bairroSelect.value = "Moema";
        }
      }

      bairroSelect.dispatchEvent(new Event("change"));
    } catch (err) {
      console.error(err);
      bairroHint.textContent = "Falha ao carregar lista de bairros do servidor.";
    }
  }

  // Carrega Resumo de Mercado da API
  async function loadMarketSummary() {
    try {
      const res = await fetch(`/api/market-summary?estado=${state.selectedUf}`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.total_registros_geral) {
        statTotalAmostras.textContent = `${data.total_registros_geral.toLocaleString("pt-BR")}`;
      }
      if (data.media_geral_m2) {
        statMediaM2.textContent = formatCurrency(data.media_geral_m2);
      }
      if (data.idade_media_imoveis) {
        statIdadeMedia.textContent = `${data.idade_media_imoveis} anos`;
      }

      // Preenche tabela de mercado
      if (data.ranking_bairros && data.ranking_bairros.length > 0) {
        marketTableBody.innerHTML = "";
        data.ranking_bairros.forEach((item) => {
          const tr = document.createElement("tr");
          const isHighLiquidity = item.preco_m2_medio >= 12000;
          const cidadeTxt = item.cidade ? `${item.cidade} — ` : "";
          
          tr.innerHTML = `
            <td><strong>${cidadeTxt}${item.bairro}</strong></td>
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

  // Carrega Resultados da Validação Cruzada K-Fold (Benchmark dos 3 Modelos)
  async function loadBenchmarkResults() {
    try {
      const res = await fetch("/api/benchmark");
      if (!res.ok) return;
      const data = await res.json();

      if (data.modelos && data.modelos.length > 0) {
        benchmarkTableBody.innerHTML = "";

        data.modelos.forEach((m, idx) => {
          const tr = document.createElement("tr");
          const isWinner = m.id === data.modelo_campeao;
          if (isWinner) tr.classList.add("benchmark-winner-row");

          tr.innerHTML = `
            <td>
              <strong>${m.nome}</strong>
              <div style="font-size: 0.72rem; color: #64748b; margin-top: 2px;">${m.descricao_didatica.substring(0, 75)}...</div>
            </td>
            <td><span class="badge-academic">${m.tipo}</span></td>
            <td>
              <strong style="color: ${isWinner ? '#16a34a' : '#1e293b'}">${m.r2_pct}%</strong> 
              <span style="font-size: 0.72rem; color: #94a3b8;">(± ${(m.r2_std * 100).toFixed(1)}%)</span>
            </td>
            <td>${formatCurrency(m.mae_medio)}</td>
            <td><strong>${m.mape_medio_pct}%</strong></td>
            <td>${m.tempo_treino_seg}s</td>
            <td>
              ${isWinner 
                ? '<span class="badge-rank-winner">🏆 Campeão (Melhor R²)</span>' 
                : `<span class="badge-rank-sub">${idx + 1}º Lugar</span>`}
            </td>
          `;
          benchmarkTableBody.appendChild(tr);
        });

        // Atualiza métrica global R² se disponível
        const winner = data.modelos.find((m) => m.id === data.modelo_campeao);
        if (winner) {
          statR2Score.textContent = `${winner.r2_pct}%`;
        }
      }
    } catch (err) {
      console.warn("Falha ao carregar benchmark:", err);
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

    const selectedBairroObj = state.bairrosMap[bairro];
    const cidade = selectedBairroObj ? selectedBairroObj.cidade : null;

    const payload = {
      estado: state.selectedUf,
      cidade: cidade,
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
    const localizacaoTxt = imovel.cidade && imovel.cidade !== "São Paulo" 
      ? `${imovel.tipo} em ${imovel.bairro} (${imovel.cidade} - ${imovel.estado})`
      : `${imovel.tipo} em ${imovel.bairro} (${imovel.estado})`;
    
    resImovelTitulo.textContent = localizacaoTxt;
    resImovelDetalhes.textContent = `${imovel.area_m2} m² • ${imovel.quartos} Dorm. • ${imovel.banheiros} Banheiros • ${imovel.vagas} Vagas • Construído em ${imovel.ano_construcao}`;

    // Valor Central
    resValorJusto.textContent = formatCurrency(prec.valor_justo);
    resPrecoM2.textContent = `${formatCurrency(prec.preco_m2)}/m²`;
    
    // Tag de Diferencial do m²
    const diff = prec.diferencial_m2_pct;
    const diffSign = diff >= 0 ? `+${diff}%` : `${diff}%`;
    resDiffM2.textContent = `${diffSign} vs. média regional`;
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

  init();
});
