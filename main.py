"""
API Backend - Sistema de Precificação Imobiliária com IA
Projeto Integrador 6 - SI-PI6-2026-T012 - Sprint 2 (SP & SC)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from ml.pricing_model import engine, DATASET_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Inicializando motor de Inteligência Artificial...")
    engine.train_or_load()
    yield
    print("[API] Finalizando serviços.")

app = FastAPI(
    title="CorretorIA - Inteligência de Precificação Imobiliária (SI-PI6)",
    description="Sistema inteligente de apoio à decisão e precificação para corretores (São Paulo & Santa Catarina).",
    version="2.0.0",
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
    estado: str = Field(default="SP", description="Sigla do estado (SP ou SC)")
    cidade: Optional[str] = Field(default=None, description="Nome da cidade")
    bairro: str = Field(..., description="Nome do bairro")
    tipo: str = Field(default="Apartamento", description="Apartamento, Casa ou Cobertura")
    area_m2: float = Field(..., gt=10, lt=2000, description="Área útil em m²")
    ano_construcao: int = Field(..., ge=1950, le=2026, description="Ano de construção do imóvel")
    quartos: int = Field(default=2, ge=1, le=10, description="Número de dormitórios")
    banheiros: int = Field(default=2, ge=1, le=10, description="Número de banheiros")
    vagas: int = Field(default=1, ge=0, le=10, description="Número de vagas de garagem")
    margem_corretor_pct: Optional[float] = Field(default=6.0, ge=1.0, le=15.0, description="Percentual de comissão do consultor")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "modulo_ia": "operacional",
        "versao": "2.0.0-sprint2",
        "estados_suportados": ["SP", "SC"],
        "metricas": engine.metrics
    }


@app.get("/api/locations")
def get_locations(estado: Optional[str] = Query(default=None, description="Filtrar por UF (SP ou SC)")):
    """
    Retorna os estados suportados e os bairros disponíveis.
    Agora com São Paulo (SP) e Santa Catarina (SC) ATIVOS!
    """
    bairros_sp = engine.get_available_bairros("SP")
    bairros_sc = engine.get_available_bairros("SC")

    estados_info = [
        {
            "uf": "SP",
            "nome": "São Paulo",
            "status": "ativo",
            "total_bairros": len(bairros_sp),
            "descricao": "Base Homologada (Capital Paulista)"
        },
        {
            "uf": "SC",
            "nome": "Santa Catarina",
            "status": "ativo",
            "total_bairros": len(bairros_sc),
            "descricao": "Base Homologada (Balneário Camboriú, Floripa, Itapema, etc.)"
        },
        {
            "uf": "MG",
            "nome": "Minas Gerais",
            "status": "em_breve",
            "total_bairros": 0,
            "descricao": "Fase 3 do Roadmap (Belo Horizonte e Região)"
        },
        {
            "uf": "RJ",
            "nome": "Rio de Janeiro",
            "status": "em_breve",
            "total_bairros": 0,
            "descricao": "Fase 3 do Roadmap (Capital e Niterói)"
        }
    ]

    if estado:
        uf = estado.upper()
        if uf == "SC":
            bairros_retorno = sorted(bairros_sc, key=lambda x: f"{x.get('cidade','')}_{x['nome']}")
        else:
            bairros_retorno = sorted(bairros_sp, key=lambda x: x["nome"])
    else:
        bairros_retorno = sorted(bairros_sp + bairros_sc, key=lambda x: x["nome"])

    return {
        "estados": estados_info,
        "bairros": bairros_retorno
    }


@app.get("/api/benchmark")
def get_benchmark():
    """
    Retorna os resultados didáticos da Validação Cruzada (5-Fold)
    comparando os 3 modelos: Regressão Linear, Árvore de Decisão e Random Forest.
    """
    return engine.get_benchmark_results()


@app.post("/api/predict")
def predict_price(req: PropertyPredictionRequest):
    estado_uf = req.estado.upper()
    if estado_uf not in ["SP", "SC"]:
        raise HTTPException(
            status_code=400,
            detail=f"O estado '{req.estado}' ainda não está homologado. Estados ativos: SP e SC."
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
            cidade=req.cidade
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da precificação: {str(e)}")


@app.get("/api/market-summary")
def market_summary(estado: Optional[str] = Query(default="SP", description="Estado para filtrar métricas")):
    if not os.path.exists(DATASET_PATH):
        return {"total_amostras": 0, "media_preco_m2": 0, "ranking_bairros": []}

    df = pd.read_csv(DATASET_PATH)
    uf = (estado or "SP").upper()
    df_estado = df[df["estado"] == uf] if uf in ["SP", "SC"] else df

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
