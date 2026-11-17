"""
Suite de Testes Automatizados - CorretorIA (6 Estados: SP, SC, MG, RJ, DF, PR)
Projeto Integrador 6:
- Base multi-estado de 18.000 amostras (2013-2026)
- Hold-out 10% cru (1.800 amostras) + GridSearchCV (5-Fold) em 5 modelos
- Estimação futura interativa (2026-2035)
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_tests():
    print("[TEST 1/6] Importando engine e modelo campeão (18.000 amostras / 6 estados)...")
    from ml.pricing_model import engine
    engine.train_or_load()
    assert engine.model is not None, "Modelo não carregado"
    print(f"-> Modelo carregado: {engine.metrics.get('model_name')} (R² Teste Cru: {engine.metrics.get('r2')})")

    print("\n[TEST 2/6] Testando predições nos 6 estados (SP, SC, MG, RJ, DF, PR)...")
    casos_estados = [
        ("SP", "São Paulo", "Moema"),
        ("SC", "Balneário Camboriú", "Barra Sul"),
        ("MG", "Belo Horizonte", "Belvedere"),
        ("RJ", "Rio de Janeiro", "Leblon"),
        ("DF", "Brasília", "Noroeste"),
        ("PR", "Curitiba", "Batel"),
    ]
    for uf, cidade, bairro in casos_estados:
        res = engine.predict(
            bairro=bairro,
            tipo="Apartamento",
            area_m2=100,
            ano_construcao=2020,
            quartos=3,
            banheiros=2,
            vagas=2,
            margem_corretor_pct=6.0,
            estado=uf,
            cidade=cidade,
            ano_estimacao=2026,
        )
        assert res["precificacao"]["valor_justo"] > 0, f"Valor inválido em {uf}"
        assert res["imovel"]["estado"] == uf
        assert [p["ano"] for p in res["estimativa_futuro"]["projecoes"]] == list(range(2027, 2036))
        print(f"   • {uf} ({cidade} - {bairro}): 2026 = R$ {res['precificacao']['valor_justo']:,.2f} -> 2035 = R$ {res['estimativa_futuro']['projecoes'][-1]['valor_estimado']:,.2f}")

    print("\n[TEST 3/6] Testando seletor de ano da estimação futura (ano_estimacao=2030 vs 2026)...")
    res_2026 = engine.predict(
        bairro="Belvedere", tipo="Apartamento", area_m2=120, ano_construcao=2021,
        quartos=3, banheiros=3, vagas=2, estado="MG", cidade="Belo Horizonte", ano_estimacao=2026
    )
    res_2030 = engine.predict(
        bairro="Belvedere", tipo="Apartamento", area_m2=120, ano_construcao=2021,
        quartos=3, banheiros=3, vagas=2, estado="MG", cidade="Belo Horizonte", ano_estimacao=2030
    )
    assert res_2030["estimativa_futuro"]["ano_selecionado"] == 2030
    assert res_2030["precificacao"]["valor_justo"] > res_2026["precificacao"]["valor_justo"]
    print(f"-> Sucesso! Belvedere (MG): 2026 = R$ {res_2026['precificacao']['valor_justo']:,.2f} | 2030 = R$ {res_2030['precificacao']['valor_justo']:,.2f}")

    print("\n[TEST 4/6] Validando Hold-out 10% Cru (1.800 amostras), GridSearchCV e 5 Modelos...")
    bench = engine.get_benchmark_results()
    assert bench.get("total_amostras") == 18000, f"Esperado 18000, obtido {bench.get('total_amostras')}"
    assert bench.get("holdout_info", {}).get("teste_cru_pct") == 10
    assert bench.get("holdout_info", {}).get("amostras_teste_cru") == 1800
    modelos = bench.get("modelos", [])
    assert len(modelos) == 5, f"Esperado 5 modelos, encontrado {len(modelos)}"
    esperados = {"ridge", "decision_tree", "random_forest", "xgboost", "mlp_neural_net"}
    assert {m["id"] for m in modelos} == esperados
    for m in modelos:
        assert "melhores_parametros" in m and len(m["melhores_parametros"]) > 0
        print(f"   • {m['nome']}: R² CV={m['r2_cv_pct']}% | R² Teste Cru={m['r2_pct']}% | MAPE={m['mape_medio_pct']}%")

    print("\n[TEST 5/6] Verificando Série Anual (2013-2026) & Projeção Futura (2027-2035) para todos os 6 estados...")
    hist = bench.get("serie_anual_historica", {})
    fut = bench.get("projecao_futura_anual", {})
    for uf in ["SP", "SC", "MG", "RJ", "DF", "PR"]:
        assert len(hist.get(uf, [])) == 14, f"Esperado 14 anos (2013-2026) para {uf}"
        assert [x["ano"] for x in fut.get(uf, [])] == list(range(2027, 2036))
    print("-> Série histórica (2013-2026) e projeções futuras (2027-2035) OK para os 6 estados!")

    print("\n[TEST 6/6] Verificando endpoints da API e arquivos estáticos...")
    import main
    locs = main.get_locations()
    ativos = [e["uf"] for e in locs["estados"] if e["status"] == "ativo"]
    assert set(ativos) == {"SP", "SC", "MG", "RJ", "DF", "PR"}
    for uf in ativos:
        tr = main.get_yearly_trends(estado=uf)
        assert tr["estado"] == uf and len(tr["estimacao_futuro"]) == 9
    for fname in ["index.html", "styles.css", "app.js"]:
        fpath = os.path.join(PROJECT_ROOT, "static", fname)
        assert os.path.exists(fpath), f"{fname} ausente"
    print("-> Todos os 6 estados ativos na API e frontend estático íntegros!")

    print("\n==========================================")
    print(" TODOS OS TESTES PASSARAM COM SUCESSO! [OK]")
    print("==========================================")


if __name__ == "__main__":
    run_tests()
