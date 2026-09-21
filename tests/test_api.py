"""
Suite de Testes Rápidos - API CorretorIA
Projeto Integrador 6
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

def run_tests():
    print("[TEST 1/4] Importando engine e modelo de IA...")
    from ml.pricing_model import engine
    engine.train_or_load()
    print("-> Engine carregada com sucesso!")

    print("\n[TEST 2/4] Testando predição de imóvel em Moema...")
    res = engine.predict(
        bairro="Moema",
        tipo="Apartamento",
        area_m2=75,
        ano_construcao=2018,
        quartos=2,
        banheiros=2,
        vagas=1,
        margem_corretor_pct=6.0
    )
    assert res["precificacao"]["valor_justo"] > 0
    assert res["precificacao"]["faixa_minima"] < res["precificacao"]["valor_justo"]
    assert res["precificacao"]["faixa_maxima"] > res["precificacao"]["valor_justo"]
    assert res["honorarios_corretor"]["comissao_valor_justo"] > 0
    print(f"-> Sucesso! Valor Justo: R$ {res['precificacao']['valor_justo']:,.2f}")
    print(f"-> Faixa: R$ {res['precificacao']['faixa_minima']:,.2f} a R$ {res['precificacao']['faixa_maxima']:,.2f}")
    print(f"-> Comissão 6%: R$ {res['honorarios_corretor']['comissao_valor_justo']:,.2f}")

    print("\n[TEST 3/4] Testando predição de Casa no Tatuapé...")
    res_casa = engine.predict(
        bairro="Tatuapé",
        tipo="Casa",
        area_m2=160,
        ano_construcao=2010,
        quartos=3,
        banheiros=3,
        vagas=2,
        margem_corretor_pct=5.0
    )
    assert res_casa["precificacao"]["valor_justo"] > 0
    print(f"-> Sucesso! Casa Tatuapé: R$ {res_casa['precificacao']['valor_justo']:,.2f}")

    print("\n[TEST 4/4] Verificando integridade dos arquivos estáticos...")
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
