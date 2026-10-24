import numpy as np
import pandas as pd
import pytest
from src.core.optimizer import PortfolioOptimizer


@pytest.fixture
def base_assets_df():
    data = {
        "Ativo": [
            "Tesouro Selic",
            "Tesouro IPCA+",
            "CDB Pós-fixado",
            "Fundo DI",
            "Fundo Multimercado",
        ],
        "Classe": ["Renda Fixa", "Renda Fixa", "Renda Fixa", "Renda Fixa", "Multimercado"],
        "Retorno_Esperado": [0.1375, 0.1375, 0.1375, 0.1375, 0.1400],
        "Volatilidade_Anual": [0.005, 0.045, 0.008, 0.006, 0.060],
        "Teto_Individual": [0.30, 0.30, 0.30, 0.30, 0.15],
        "Piso_Individual": [0.00, 0.00, 0.00, 0.00, 0.00],
    }
    return pd.DataFrame(data)


def test_portfolio_budget_constraint(base_assets_df):
    """Garante que a soma de todas as alocações equivale a 100% do capital."""
    capital = 150_000.0
    optimizer = PortfolioOptimizer(ativos_df=base_assets_df, capital_total=capital)
    df_res = optimizer.otimizar()

    soma_alocada = df_res["Alocacao_R$"].sum()
    assert np.isclose(soma_alocada, capital, atol=1e-2)
    assert np.isclose(df_res["Alocacao_%"].sum(), 100.0, atol=1e-2)


def test_individual_bounds_respected(base_assets_df):
    """Garante que nenhum ativo ultrapassa o seu respetivo teto ou piso individual."""
    capital = 100_000.0
    optimizer = PortfolioOptimizer(ativos_df=base_assets_df, capital_total=capital)
    df_res = optimizer.otimizar()

    for _, row in df_res.iterrows():
        assert row["Alocacao_R$"] <= row["Teto_R$"] + 1e-4
        assert row["Alocacao_R$"] >= row["Piso_R$"] - 1e-4


def test_fixed_income_floor_constraint(base_assets_df):
    """Garante que a restrição de classe (Piso de RF) é estritamente cumprida."""
    capital = 100_000.0
    min_rf = 0.70
    optimizer = PortfolioOptimizer(ativos_df=base_assets_df, capital_total=capital, min_rf_pct=min_rf)
    df_res = optimizer.otimizar()

    rf_total = df_res[df_res["Classe"] == "Renda Fixa"]["Alocacao_R$"].sum()
    assert rf_total >= (min_rf * capital) - 1e-4


def test_stage2_quadratic_tie_breaking(base_assets_df):
    """Garante que o desempate por QP reduz ou mantém a variância entre soluções ótimas de LP."""
    opt_with_qp = PortfolioOptimizer(ativos_df=base_assets_df, capital_total=100_000.0, minimizar_risco_no_empate=True)
    opt_with_qp.otimizar()
    m_qp = opt_with_qp.calcular_metricas_risco()

    opt_no_qp = PortfolioOptimizer(ativos_df=base_assets_df, capital_total=100_000.0, minimizar_risco_no_empate=False)
    opt_no_qp.otimizar()
    m_no_qp = opt_no_qp.calcular_metricas_risco()

    assert np.isclose(m_qp["Retorno_Anual_%"], m_no_qp["Retorno_Anual_%"], atol=1e-5)
    assert m_qp["Volatilidade_Anual_%"] <= m_no_qp["Volatilidade_Anual_%"] + 1e-6


def test_shadow_prices_consistency(base_assets_df):
    """Garante que o preço-sombra do capital e do ativo de maior retorno são positivos."""
    optimizer = PortfolioOptimizer(ativos_df=base_assets_df, capital_total=100_000.0)
    optimizer.otimizar()

    assert optimizer.precos_sombra["capital"] > 0.0

    mm_row = optimizer.df[optimizer.df["Ativo"] == "Fundo Multimercado"].iloc[0]
    assert np.isclose(mm_row["Preco_Sombra_Teto"], 0.0025, atol=1e-4)


def test_infeasible_inputs_raise_error(base_assets_df):
    """Garante que limites inconsistentes disparam erro antes de submeter ao solver."""
    df_invalido = base_assets_df.copy()
    df_invalido.loc[0, "Teto_Individual"] = 0.05
    df_invalido.loc[0, "Piso_Individual"] = 0.10

    with pytest.raises(ValueError):
        PortfolioOptimizer(ativos_df=df_invalido)