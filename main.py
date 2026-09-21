"""
API Backend - Sistema de Precificação Imobiliária com IA
Projeto Integrador 6 - SI-PI6-2026-T012
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from ml.pricing_model import engine, DATASET_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializa e carrega o modelo preditivo
    print("[API] Inicializando motor de Inteligência Artificial...")
    engine.train_or_load()
    yield
    print("[API] Finalizando serviços.")

app = FastAPI(
    title="CorretorIA - Inteligência de Precificação Imobiliária (SI-PI6)",
    description="Sistema inteligente de apoio à decisão e precificação para corretores de imóveis.",
    version="1.0.0",
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
    estado: str = Field(default="SP", description="Sigla do estado (Ex: SP)")
    bairro: str = Field(..., description="Nome do bairro em São Paulo")
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
        "versao": "1.0.0-sprint1",
        "metricas": engine.metrics
    }


@app.get("/api/locations")
def get_locations():
    """
    Retorna os estados suportados e os bairros disponíveis.
    Demonstra a arquitetura multi-estado com SP ativo e MG/SC no roadmap.
    """
    bairros_sp = engine.get_available_bairros()
    return {
        "estados": [
            {
                "uf": "SP",
                "nome": "São Paulo",
                "status": "ativo",
                "total_bairros": len(bairros_sp),
                "descricao": "Base Piloto Ativa - Modelo Preditivo Homologado"
            },
            {
                "uf": "MG",
                "nome": "Minas Gerais",
                "status": "em_breve",
                "total_bairros": 0,
                "descricao": "Fase 2 do Roadmap (Belo Horizonte e Região)"
            },
            {
                "uf": "SC",
                "nome": "Santa Catarina",
                "status": "em_breve",
                "total_bairros": 0,
                "descricao": "Fase 2 do Roadmap (Florianópolis, Balneário Camboriú)"
            },
            {
                "uf": "RJ",
                "nome": "Rio de Janeiro",
                "status": "em_breve",
                "total_bairros": 0,
                "descricao": "Fase 3 do Roadmap (Capital e Niterói)"
            }
        ],
        "bairros": sorted(bairros_sp, key=lambda x: x["nome"])
    }


@app.post("/api/predict")
def predict_price(req: PropertyPredictionRequest):
    if req.estado.upper() != "SP":
        raise HTTPException(
            status_code=400,
            detail=f"O estado '{req.estado}' está planejado para a Fase 2 de expansão. No momento, o modelo atende São Paulo (SP)."
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
            estado=req.estado.upper()
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da precificação: {str(e)}")


@app.get("/api/market-summary")
def market_summary():
    """
    Retorna métricas consolidadas do mercado de São Paulo para o dashboard do corretor.
    """
    if not os.path.exists(DATASET_PATH):
        return {"total_amostras": 0, "media_preco_m2": 0, "ranking_bairros": []}

    df = pd.read_csv(DATASET_PATH)
    resumo_bairros = df.groupby("bairro").agg(
        preco_m2_medio=("preco_m2", "mean"),
        preco_medio=("preco", "mean"),
        amostras=("id", "count")
    ).reset_index()

    resumo_bairros["preco_m2_medio"] = resumo_bairros["preco_m2_medio"].round(2)
    resumo_bairros["preco_medio"] = resumo_bairros["preco_medio"].round(2)
    resumo_bairros = resumo_bairros.sort_values(by="preco_m2_medio", ascending=False)

    return {
        "total_registros_base": len(df),
        "media_geral_m2": round(float(df["preco_m2"].mean()), 2),
        "idade_media_imoveis": round(float(df["idade_imovel"].mean()), 1),
        "ranking_bairros": resumo_bairros.to_dict(orient="records")
    }


# Monta frontend estático
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
