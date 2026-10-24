import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Portfolio Optimization Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Asset Allocation & Portfolio Optimization")
st.markdown("Otimização em **dois estágios (HiGHS LP + Mínima Variância)** com análise de sensibilidade e preços-sombra.")

API_URL = st.sidebar.text_input("URL da API FastAPI", value="http://localhost:8000")

st.sidebar.header("⚙️ Parâmetros Operacionais")
capital = st.sidebar.number_input("Capital Total (R$)", value=100_000.0, step=10_000.0)
min_rf = st.sidebar.slider("Piso de Renda Fixa (%)", min_value=10, max_value=90, value=60) / 100.0
taxa_livre = st.sidebar.number_input("Taxa Livre de Risco (% a.a.)", value=13.75, step=0.25) / 100.0
desempate_risco = st.sidebar.checkbox("Desempatar por Mínima Variância", value=True)

st.subheader("📋 Universo de Ativos e Limites")
dados_iniciais = [
    {"Ativo": "Tesouro Selic", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.005, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "Tesouro IPCA+", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.045, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "CDB Pós-fixado", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.008, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "Fundo DI", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.006, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "Fundo Multimercado", "Classe": "Multimercado", "Retorno_Esperado": 0.1400, "Volatilidade_Anual": 0.060, "Teto_Individual": 0.15, "Piso_Individual": 0.0},
]

df_editado = st.data_editor(pd.DataFrame(dados_iniciais), num_rows="dynamic", use_container_width=True)

if st.button("🚀 Otimizar Carteira", type="primary"):
    payload = {
        "ativos": df_editado.to_dict(orient="records"),
        "capital_total": capital,
        "min_rf_pct": min_rf,
        "taxa_livre_risco": taxa_livre,
        "corr_base": 0.85,
        "corr_multimercado": 0.25,
        "minimizar_risco_no_empate": desempate_risco
    }
    
    with st.spinner("Resolvendo modelos LP e QP..."):
        try:
            res_opt = requests.post(f"{API_URL}/api/v1/optimize", json=payload)
            if res_opt.status_code != 200:
                st.error(f"Erro na API: {res_opt.json().get('detail', 'Erro desconhecido')}")
            else:
                dados = res_opt.json()
                metricas = dados["metricas"]
                alocacao = pd.DataFrame(dados["alocacao"])
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno Anual", f"{metricas['Retorno_Anual_%']:.2f}% a.a.")
                c2.metric("Volatilidade Estimada", f"{metricas['Volatilidade_Anual_%']:.2f}% a.a.")
                sharpe_val = f"{metricas['Indice_Sharpe']:.2f}" if metricas["Indice_Sharpe"] is not None else "N/A"
                c3.metric("Índice de Sharpe", sharpe_val)
                c4.metric("Preço-Sombra Capital", f"{dados['precos_sombra']['capital']*100:.2f}% a.a.")
                
                st.markdown("### 📌 Alocação Ótima e Preços-Sombra")
                st.dataframe(alocacao[[
                    "Ativo", "Classe", "Retorno_Esperado", "Volatilidade_Anual",
                    "Alocacao_R$", "Alocacao_%", "Folga_Teto_R$", "Preco_Sombra_Teto"
                ]], use_container_width=True)
                
                st.markdown("### 📈 Dashboard Analítico")
                res_plot = requests.post(f"{API_URL}/api/v1/plot", json=payload)
                if res_plot.status_code == 200:
                    img_data = res_plot.json()["image_base64"]
                    st.image(f"data:image/png;base64,{img_data}", use_container_width=True)
                    
        except requests.exceptions.ConnectionError:
            st.error(f"Não foi possível conectar ao backend em `{API_URL}`. Certifique-se de que o Uvicorn está rodando.")