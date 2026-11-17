"""
API Backend - Sistema de Precificação Imobiliária com IA (6 Estados)
Projeto Integrador 6 - SI-PI6-2026-T012
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from ml.pricing_model import engine, DATASET_PATH

ESTADOS_ATIVOS = ["SP", "SC", "MG", "RJ", "DF", "PR"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Inicializando motor de Inteligência Artificial...")
    engine.train_or_load()
    yield
    print("[API] Finalizando serviços.")


app = FastAPI(
    title="CorretorIA - Inteligência de Precificação Imobiliária (SI-PI6)",
    description="Sistema inteligente de apoio à decisão, precificação e estimação futura em 6 estados brasileiros.",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PropertyPredictionRequest(BaseModel):
    estado: str = Field(default="SP", description="Sigla do estado (SP, SC, MG, RJ, DF ou PR)")
    cidade: Optional[str] = Field(default=None, description="Nome da cidade")
    bairro: str = Field(..., description="Nome do bairro")
    tipo: str = Field(default="Apartamento", description="Apartamento, Casa ou Cobertura")
    area_m2: float = Field(..., gt=10, lt=2000, description="Área útil em m²")
    ano_construcao: int = Field(..., ge=1950, le=2026, description="Ano de construção do imóvel")
    quartos: int = Field(default=2, ge=1, le=10, description="Número de dormitórios")
    banheiros: int = Field(default=2, ge=1, le=10, description="Número de banheiros")
    vagas: int = Field(default=1, ge=0, le=10, description="Número de vagas de garagem")
    margem_corretor_pct: Optional[float] = Field(default=6.0, ge=1.0, le=15.0, description="Percentual de comissão do consultor")
    ano_estimacao: int = Field(default=2026, ge=2026, le=2035, description="Ano alvo para precificação/estimação futura (2026 a 2035)")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "modulo_ia": "operacional",
        "versao": "3.1.0-multiestado",
        "estados_suportados": ESTADOS_ATIVOS,
        "metricas": engine.metrics
    }


@app.get("/api/locations")
def get_locations(estado: Annotated[Optional[str], Query(description="Filtrar por UF (SP, SC, MG, RJ, DF, PR)")] = None):
    """
    Retorna os 6 estados homologados e os bairros disponíveis.
    """
    estados_meta = [
        ("SP", "São Paulo", "Capital Paulista"),
        ("SC", "Santa Catarina", "Balneário Camboriú, Floripa, Itapema, Joinville, Blumenau"),
        ("MG", "Minas Gerais", "Belo Horizonte e Uberlândia"),
        ("RJ", "Rio de Janeiro", "Rio de Janeiro e Niterói"),
        ("DF", "Distrito Federal", "Brasília (Plano Piloto e Águas Claras)"),
        ("PR", "Paraná", "Curitiba e Londrina"),
    ]

    estados_info = []
    todos_bairros = []
    for uf, nome, desc in estados_meta:
        b_list = engine.get_available_bairros(uf)
        todos_bairros.extend(b_list)
        estados_info.append({
            "uf": uf,
            "nome": nome,
            "status": "ativo",
            "total_bairros": len(b_list),
            "descricao": f"Base Homologada ({desc})"
        })

    if estado:
        uf = estado.upper()
        if uf not in ESTADOS_ATIVOS:
            raise HTTPException(status_code=400, detail=f"Estado '{estado}' não suportado. Estados ativos: {', '.join(ESTADOS_ATIVOS)}.")
        bairros_retorno = sorted(engine.get_available_bairros(uf), key=lambda x: f"{x.get('cidade', '')}_{x['nome']}")
    else:
        bairros_retorno = sorted(todos_bairros, key=lambda x: f"{x.get('estado', '')}_{x['nome']}")

    return {
        "estados": estados_info,
        "bairros": bairros_retorno
    }


@app.get("/api/benchmark")
def get_benchmark():
    """
    Retorna os resultados do GridSearchCV (5-Fold) nos 90% de treino
    e métricas avaliadas no conjunto de teste cru (10% isolado) para os 5 modelos.
    """
    return engine.get_benchmark_results()


@app.get("/api/yearly-trends")
def get_yearly_trends(estado: Annotated[Optional[str], Query(description="Filtrar por UF (SP, SC, MG, RJ, DF, PR)")] = None):
    """
    Retorna a série histórica anual (2013-2026) e a estimação de futuro (2027-2035)
    para os 6 estados ativos.
    """
    bench = engine.get_benchmark_results()
    hist = bench.get("serie_anual_historica", {})
    fut = bench.get("projecao_futura_anual", {})

    if estado:
        uf = estado.upper()
        if uf not in ESTADOS_ATIVOS:
            raise HTTPException(status_code=400, detail=f"Estado '{estado}' não suportado.")
        return {
            "estado": uf,
            "historico_anual": hist.get(uf, []),
            "estimacao_futuro": fut.get(uf, [])
        }

    return {
        "historico_anual": hist,
        "estimacao_futuro": fut
    }


@app.post("/api/predict")
def predict_price(req: PropertyPredictionRequest):
    estado_uf = req.estado.upper()
    if estado_uf not in ESTADOS_ATIVOS:
        raise HTTPException(
            status_code=400,
            detail=f"O estado '{req.estado}' ainda não está homologado. Estados ativos: {', '.join(ESTADOS_ATIVOS)}."
        )

    try:
        resultado = engine.predict(
            bairro=req.bairro,
            tipo=req.tipo,
            area_m2=req.area_m2,
            ano_construcao=req.ano_construcao,
            quartos=req.quartos,
            banheiros=req.banheiros,
            vagas=req.vagas,
            margem_corretor_pct=req.margem_corretor_pct or 6.0,
            estado=estado_uf,
            cidade=req.cidade,
            ano_estimacao=req.ano_estimacao
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da precificação: {str(e)}")


@app.get("/api/market-summary")
def market_summary(estado: Annotated[Optional[str], Query(description="Estado para filtrar métricas")] = "SP"):
    if not os.path.exists(DATASET_PATH):
        return {"total_amostras": 0, "media_preco_m2": 0, "ranking_bairros": []}

    df = pd.read_csv(DATASET_PATH)
    uf = (estado or "SP").upper()
    df_estado = df[df["estado"] == uf] if uf in ESTADOS_ATIVOS else df

    resumo_bairros = df_estado.groupby(["cidade", "bairro"]).agg(
        preco_m2_medio=("preco_m2", "mean"),
        preco_medio=("preco", "mean"),
        amostras=("id", "count")
    ).reset_index()

    resumo_bairros["preco_m2_medio"] = resumo_bairros["preco_m2_medio"].round(2)
    resumo_bairros["preco_medio"] = resumo_bairros["preco_medio"].round(2)
    resumo_bairros = resumo_bairros.sort_values(by="preco_m2_medio", ascending=False)

    return {
        "estado": uf,
        "total_registros_base": len(df_estado),
        "total_registros_geral": len(df),
        "media_geral_m2": round(float(df_estado["preco_m2"].mean()), 2),
        "idade_media_imoveis": round(float(df_estado["idade_imovel"].mean()), 1),
        "ranking_bairros": resumo_bairros.to_dict(orient="records")
    }


# Monta frontend estático
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
