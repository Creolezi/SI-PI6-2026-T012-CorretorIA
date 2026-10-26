"""
Suite de Testes Rápidos - API CorretorIA (SP & SC)
Projeto Integrador 6 - Sprint 2
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

def run_tests():
    print("[TEST 1/5] Importando engine e modelo de IA campeão...")
    from ml.pricing_model import engine
    engine.train_or_load()
    print("-> Engine carregada com sucesso!")

    print("\n[TEST 2/5] Testando predição de imóvel em São Paulo (Moema)...")
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
    assert res_sp["honorarios_corretor"]["comissao_valor_justo"] > 0
    print(f"-> Sucesso! SP Moema Valor Justo: R$ {res_sp['precificacao']['valor_justo']:,.2f}")
    print(f"-> Faixa: R$ {res_sp['precificacao']['faixa_minima']:,.2f} a R$ {res_sp['precificacao']['faixa_maxima']:,.2f}")

    print("\n[TEST 3/5] Testando predição em Santa Catarina (Balneário Camboriú)...")
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
    print(f"-> Sucesso! SC Barra Sul (BC) Valor Justo: R$ {res_sc['precificacao']['valor_justo']:,.2f}")
    print(f"-> Preço do m²: R$ {res_sc['precificacao']['preco_m2']:,.2f}/m²")

    print("\n[TEST 4/5] Verificando relatório de Benchmark dos 3 modelos (K-Fold 5)...")
    benchmark_data = engine.get_benchmark_results()
    assert "modelos" in benchmark_data, "Modelos ausentes no benchmark"
    assert len(benchmark_data["modelos"]) == 3, f"Esperado 3 modelos, obtido {len(benchmark_data['modelos'])}"
    print(f"-> Sucesso! {len(benchmark_data['modelos'])} modelos avaliados com K-Fold 5.")
    for m in benchmark_data["modelos"]:
        print(f"   • {m['nome']}: R² = {m['r2_pct']}% | MAPE = {m['mape_medio_pct']}%")

    print("\n[TEST 5/5] Verificando integridade dos arquivos estáticos...")
    index_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    css_path = os.path.join(PROJECT_ROOT, "static", "styles.css")
    js_path = os.path.join(PROJECT_ROOT, "static", "app.js")
    assert os.path.exists(index_path), "index.html ausente"
    assert os.path.exists(css_path), "styles.css ausente"
    assert os.path.exists(js_path), "app.js ausente"
    print("-> Todos os arquivos do frontend estático estão presentes e íntegros!")

    print("\n==========================================")
    print(" TODOS OS TESTES PASSARAM COM SUCESSO! [OK]")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
