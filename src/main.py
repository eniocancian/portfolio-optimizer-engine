import base64
import io
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importação modular a partir do pacote core
from src.core.optimizer import PortfolioOptimizer

app = FastAPI(
    title="Portfolio Optimization Engine API",
    description="API para Otimização de Carteiras em 2 Estágios (LP HiGHS + QP Min Variance)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AtivoItem(BaseModel):
    Ativo: str
    Classe: str
    Retorno_Esperado: float = Field(..., gt=-1.0, lt=2.0)
    Volatilidade_Anual: float = Field(..., ge=0.0)
    Teto_Individual: float = Field(..., gt=0.0, le=1.0)
    Piso_Individual: float = Field(default=0.0, ge=0.0, le=1.0)

class OptimizationRequest(BaseModel):
    ativos: List[AtivoItem]
    capital_total: float = Field(default=100_000.0, gt=0)
    min_rf_pct: float = Field(default=0.60, ge=0.0, le=1.0)
    taxa_livre_risco: float = Field(default=0.1375, ge=0.0)
    corr_base: float = 0.85
    corr_multimercado: float = 0.25
    minimizar_risco_no_empate: bool = True

class SensitivityRequest(OptimizationRequest):
    ativo_alvo: Optional[str] = None
    teto_min: float = 0.0
    teto_max: float = 0.60
    passos: int = 25

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "portfolio-optimizer-api"}

@app.post("/api/v1/optimize")
def optimize_portfolio(payload: OptimizationRequest):
    try:
        df_ativos = pd.DataFrame([a.model_dump() for a in payload.ativos])
        opt = PortfolioOptimizer(
            ativos_df=df_ativos,
            capital_total=payload.capital_total,
            min_rf_pct=payload.min_rf_pct,
            corr_base=payload.corr_base,
            corr_multimercado=payload.corr_multimercado,
            minimizar_risco_no_empate=payload.minimizar_risco_no_empate
        )

        df_res = opt.otimizar()
        metricas = opt.calcular_metricas_risco(taxa_livre_risco=payload.taxa_livre_risco)

        if np.isnan(metricas["Indice_Sharpe"]):
            metricas["Indice_Sharpe"] = None

        return {
            "status": "success",
            "alocacao": df_res.to_dict(orient="records"),
            "metricas": metricas,
            "precos_sombra": opt.precos_sombra
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/sensitivity")
def compute_sensitivity(payload: SensitivityRequest):
    try:
        df_ativos = pd.DataFrame([a.model_dump() for a in payload.ativos])
        opt = PortfolioOptimizer(
            ativos_df=df_ativos,
            capital_total=payload.capital_total,
            min_rf_pct=payload.min_rf_pct,
            corr_base=payload.corr_base,
            corr_multimercado=payload.corr_multimercado
        )
        opt.otimizar()
        sens_df = opt.simular_sensibilidade(
            nome_ativo=payload.ativo_alvo,
            teto_min=payload.teto_min,
            teto_max=payload.teto_max,
            passos=payload.passos
        )
        return {
            "status": "success",
            "ativo_sensibilidade": opt._ativo_sens,
            "curva": sens_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/plot")
def generate_plot(payload: OptimizationRequest):
    try:
        df_ativos = pd.DataFrame([a.model_dump() for a in payload.ativos])
        opt = PortfolioOptimizer(
            ativos_df=df_ativos,
            capital_total=payload.capital_total,
            min_rf_pct=payload.min_rf_pct,
            corr_base=payload.corr_base,
            corr_multimercado=payload.corr_multimercado
        )
        opt.otimizar()
        fig = opt.plotar_dashboard(mostrar=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {"status": "success", "image_base64": img_base64}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))