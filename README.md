# 📈 Two-Stage Portfolio Optimizer Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-8CAAE6.svg)](https://scipy.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://pytest.org/)

Motor quantitativo profissional de alocação de ativos em **dois estágios**, desenvolvido para solucionar analiticamente o problema de **degenerescência do método Simplex** em carteiras de investimentos com ativos de taxas idênticas.

O projeto é disponibilizado como uma solução completa de engenharia de software: **núcleo matemático desacoplado**, **API REST em FastAPI** com validação de esquemas via Pydantic e **frontend analítico em Streamlit**.

---

## 🔬 Formulação Matemática & Solução em Dois Estágios

Em problemas reais de alocação de capital (especialmente em mercados de Renda Fixa com múltiplos instrumentos indexados ao CDI/Selic), ativos distintos frequentemente possuem o mesmo retorno esperado. Na **Programação Linear (LP)** clássica, isso gera degenerescência: infinitas combinações no hiperplano de restrições produzem a mesma função objetivo, fazendo com que o solver Simplex retorne um vértice arbitrário, contaminando métricas de risco como volatilidade e Índice de Sharpe.

Para garantir otimização global e estabilidade, este motor implementa uma esteira sequencial:

Universo de Ativos & Limites ]
│
▼
┌───────────────────────────┐
│ Estágio 1: LP (HiGHS)     │ ──> Maximiza retorno sujeito a orçamento,
│                           │     piso de classe e tetos individuais.
└───────────────────────────┘     Extrai Preços-Sombra (Variáveis Duais).
│
│ Retorno Ótimo (R*)
▼
┌───────────────────────────┐
│ Estágio 2: QP (trust-c.)  │ ──> Minimiza wᵀΣw (variância da carteira)
│                           │     travando o retorno em rᵀw ≥ R* - ε.
└───────────────────────────┘
│
▼
[ Alocação Ótima Estável (Menor Risco para o Máximo Retorno) ]

### 1. Estágio 1: Programação Linear (LP via HiGHS)
Maximiza-se o retorno esperado financeiro sujeito às restrições regulatórias e de política:

$$\max_{\mathbf{x}} \quad \mathbf{r}^T \mathbf{x}$$

Sujeito a:
* $\sum_{i=1}^n x_i = C$ (Restrição de Orçamento Total)
* $\sum_{i \in \text{RF}} x_i \ge \alpha \cdot C$ (Piso regulatório/política de Renda Fixa)
* $L_i \le x_i \le U_i, \quad \forall i \in \{1, \dots, n\}$ (Limites inferiores e superiores por ativo)

O dual do problema devolve os **preços-sombra** (custos de oportunidade):
* **Capital**: Ganho marginal por unidade monetária adicional de patrimônio líquido.
* **Tetos/Pisos**: Taxa marginal de retorno obtida ao afrouxar os limites individuais.

### 2. Estágio 2: Desempate por Risco Mínimo (QP via `trust-constr`)
Com o retorno ótimo $R^* = \mathbf{r}^T \mathbf{x}^*_{\text{LP}}$ garantido pelo Estágio 1, formula-se um problema de Programação Quadrática com restrições lineares sobre os pesos normalizados $\mathbf{w} = \mathbf{x} / C$:

$$\min_{\mathbf{w}} \quad \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$

Sujeito a:
* $\mathbf{r}^T \mathbf{w} \ge R^* - \epsilon$ (Garantia de retorno ótimo do Estágio 1)
* $\sum_{i=1}^n w_i = 1$
* $\sum_{i \in \text{RF}} w_i \ge \alpha$
* $\frac{L_i}{C} \le w_i \le \frac{U_i}{C}$

Onde $\mathbf{\Sigma}$ é a matriz de covariância estimada dos ativos. A utilização do algoritmo `trust-constr` com gradientes analíticos assegura a convergência partindo do ponto inicial fornecido pelo LP, eliminando o ruído de vértices arbitrários.

---

## 🏛️ Arquitetura do Repositório

```text
portfolio-optimizer-engine/
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # API REST (FastAPI) e rotas de cálculo
│   └── core/
│       ├── __init__.py
│       └── optimizer.py         # Classe matemática PortfolioOptimizer (LP + QP)
│
├── frontend/
│   └── app.py                   # Dashboard web interativo (Streamlit)
│
├── tests/
│   └── test_optimizer.py        # Suíte de testes automatizados (pytest)
│
├── data/                        # Dados de entrada e configurações
├── requirements.txt             # Dependências versionadas do projeto
├── .gitignore                   # Exclusões de ambiente e arquivos locais
└── README.md                    # Documentação técnica do sistema
⚡ Guia de Execução Rápida
1. Clonar o projeto e instalar dependências
Bash
git clone [https://github.com/eniocancian/portfolio-optimizer-engine.git](https://github.com/eniocancian/portfolio-optimizer-engine.git)
cd portfolio-optimizer-engine
pip install -r requirements.txt
2. Executar os testes automatizados
PowerShell
$env:PYTHONPATH="."
python -m pytest tests/ -v
3. Iniciar o Backend (FastAPI)
Bash
uvicorn src.main:app --reload --port 8000
Swagger UI / Documentação Interativa: http://127.0.0.1:8000/docs

Health Check: http://127.0.0.1:8000/health

4. Iniciar o Frontend (Streamlit)
Em outro terminal:

Bash
streamlit run frontend/app.py
Acesse o dashboard no navegador em: http://localhost:8501

📡 Contratos da API (FastAPI)
POST /api/v1/optimize
Processa a otimização em dois estágios e retorna a alocação de equilíbrio, métricas de risco e preços-sombra.

Exemplo de Payload:

JSON
{
  "ativos": [
    {"Ativo": "Tesouro Selic", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.005, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "Tesouro IPCA+", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.045, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "CDB Pós-fixado", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.008, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "Fundo DI", "Classe": "Renda Fixa", "Retorno_Esperado": 0.1375, "Volatilidade_Anual": 0.006, "Teto_Individual": 0.30, "Piso_Individual": 0.0},
    {"Ativo": "Fundo Multimercado", "Classe": "Multimercado", "Retorno_Esperado": 0.1400, "Volatilidade_Anual": 0.060, "Teto_Individual": 0.15, "Piso_Individual": 0.0}
  ],
  "capital_total": 100000.0,
  "min_rf_pct": 0.60,
  "taxa_livre_risco": 0.1375,
  "minimizar_risco_no_empate": true
}

OST /api/v1/sensitivity
Varia dinamicamente o teto de um ativo para traçar o trade-off retorno vs. risco e identificar quebras de restrição.

POST /api/v1/plot
Retorna os quatro gráficos do painel de diagnóstico codificados em Base64 para consumo direto por interfaces clientes.

🛠️ Stack Tecnológica
Modelagem & Otimização: SciPy (scipy.optimize.linprog com HiGHS, scipy.optimize.minimize com trust-constr), NumPy, Pandas.

Backend: FastAPI, Uvicorn, Pydantic v2.

Frontend: Streamlit, Matplotlib.

Qualidade de Software: Pytest, PEP-8 compliance.

👤 Autor
Desenvolvido por Ênio Fernando Cancian Júnior.

GitHub: @eniocancian

