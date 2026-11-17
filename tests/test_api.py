"""
Suite de Testes Automatizados - CorretorIA (SP & SC)
Projeto Integrador 6 - Sprint 3:
- Hold-out 10% cru
- GridSearchCV (5-Fold) em 5 modelos (Ridge, Decision Tree, Random Forest, XGBoost, MLP)
- Série Anual (2013-2026) & Estimação de Futuro (2027-2030)
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


def run_tests():
    print("[TEST 1/6] Importando engine e modelo campeão do GridSearchCV...")
    from ml.pricing_model import engine
    engine.train_or_load()
    assert engine.model is not None, "Modelo não carregado"
    print(f"-> Modelo carregado: {engine.metrics.get('model_name')} (R² Teste Cru: {engine.metrics.get('r2')})")

    print("\n[TEST 2/6] Testando predição e estimação futura em São Paulo (Moema)...")
    res_sp = engine.predict(
        bairro="Moema",
        tipo="Apartamento",
        area_m2=75,
        ano_construcao=2018,
        quartos=2,
        banheiros=2,
        vagas=1,
        margem_corretor_pct=6.0,
        estado="SP"
    )
    assert res_sp["precificacao"]["valor_justo"] > 0
    assert res_sp["precificacao"]["faixa_minima"] < res_sp["precificacao"]["valor_justo"]
    assert res_sp["precificacao"]["faixa_maxima"] > res_sp["precificacao"]["valor_justo"]
    assert "estimativa_futuro" in res_sp
    proj_sp = res_sp["estimativa_futuro"]["projecoes"]
    assert [p["ano"] for p in proj_sp] == [2027, 2028, 2029, 2030]
    assert proj_sp[-1]["valor_estimado"] > res_sp["precificacao"]["valor_justo"]
    print(f"-> Sucesso! SP Moema 2026: R$ {res_sp['precificacao']['valor_justo']:,.2f} -> 2030: R$ {proj_sp[-1]['valor_estimado']:,.2f}")

    print("\n[TEST 3/6] Testando predição e estimação futura em Santa Catarina (Balneário Camboriú)...")
    res_sc = engine.predict(
        bairro="Barra Sul",
        tipo="Apartamento",
        area_m2=130,
        ano_construcao=2021,
        quartos=3,
        banheiros=3,
        vagas=2,
        margem_corretor_pct=6.0,
        estado="SC",
        cidade="Balneário Camboriú"
    )
    assert res_sc["precificacao"]["valor_justo"] > 0
    assert res_sc["imovel"]["estado"] == "SC"
    proj_sc = res_sc["estimativa_futuro"]["projecoes"]
    assert [p["ano"] for p in proj_sc] == [2027, 2028, 2029, 2030]
    print(f"-> Sucesso! SC Barra Sul 2026: R$ {res_sc['precificacao']['valor_justo']:,.2f} -> 2030: R$ {proj_sc[-1]['valor_estimado']:,.2f}")

    print("\n[TEST 4/6] Validando Hold-out 10% Cru, GridSearchCV e os 5 Modelos no Benchmark...")
    bench = engine.get_benchmark_results()
    assert bench.get("holdout_info", {}).get("teste_cru_pct") == 10
    assert bench.get("holdout_info", {}).get("treino_cv_pct") == 90
    modelos = bench.get("modelos", [])
    assert len(modelos) == 5, f"Esperado 5 modelos, encontrado {len(modelos)}"
    ids_presentes = {m["id"] for m in modelos}
    esperados = {"ridge", "decision_tree", "random_forest", "xgboost", "mlp_neural_net"}
    assert ids_presentes == esperados, f"IDs divergentes: {ids_presentes}"
    for m in modelos:
        assert "melhores_parametros" in m and len(m["melhores_parametros"]) > 0
        assert "r2_cv_pct" in m and "r2_pct" in m and "mae_medio" in m and "mape_medio_pct" in m
        print(f"   • {m['nome']}: R² CV={m['r2_cv_pct']}% | R² Teste Cru={m['r2_pct']}% | MAPE={m['mape_medio_pct']}%")

    print("\n[TEST 5/6] Verificando Série Anual Histórica (2013-2026) & Projeção Futura (2027-2030)...")
    hist = bench.get("serie_anual_historica", {})
    fut = bench.get("projecao_futura_anual", {})
    for uf in ["SP", "SC"]:
        assert len(hist.get(uf, [])) == 14, f"Esperado 14 anos (2013-2026) para {uf}"
        assert [x["ano"] for x in fut.get(uf, [])] == [2027, 2028, 2029, 2030]
    print("-> Série histórica (2013-2026) e projeções (2027-2030) OK para SP e SC!")

    print("\n[TEST 6/6] Verificando endpoints da API e arquivos estáticos...")
    import main
    trends_sc = main.get_yearly_trends(estado="SC")
    assert trends_sc["estado"] == "SC" and len(trends_sc["estimacao_futuro"]) == 4
    for fname in ["index.html", "styles.css", "app.js"]:
        fpath = os.path.join(PROJECT_ROOT, "static", fname)
        assert os.path.exists(fpath), f"{fname} ausente"
    print("-> Endpoints e frontend estático íntegros!")

    print("\n==========================================")
    print(" TODOS OS TESTES PASSARAM COM SUCESSO! [OK]")
    print("==========================================")


if __name__ == "__main__":
    run_tests()
