import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

COLUNAS_OBRIGATORIAS = {
    "Ativo", "Classe", "Retorno_Esperado", "Volatilidade_Anual",
    "Teto_Individual", "Piso_Individual",
}

class PortfolioOptimizer:
    CLASSE_RF = "Renda Fixa"
    CLASSE_MM = "Multimercado"
    CORES_CLASSES = {"Renda Fixa": "#1f77b4", "Multimercado": "#d62728"}
    TOL_RETORNO = 1e-12

    def __init__(
        self,
        ativos_df: pd.DataFrame,
        capital_total: float = 100_000.0,
        min_rf_pct: float = 0.60,
        corr_base: float = 0.85,
        corr_multimercado: float = 0.25,
        matriz_correlacao: np.ndarray = None,
        minimizar_risco_no_empate: bool = True,
    ):
        faltando = COLUNAS_OBRIGATORIAS - set(ativos_df.columns)
        if faltando:
            raise ValueError(f"Colunas ausentes em ativos_df: {sorted(faltando)}")

        self.df = ativos_df.copy().reset_index(drop=True)
        self.capital_total = float(capital_total)
        self.min_rf_pct = float(min_rf_pct)
        self.corr_base = float(corr_base)
        self.corr_multimercado = float(corr_multimercado)
        self.minimizar_risco_no_empate = minimizar_risco_no_empate

        self.df["Teto_R$"] = self.df["Teto_Individual"] * self.capital_total
        self.df["Piso_R$"] = self.df["Piso_Individual"] * self.capital_total
        self._validar_limites()

        self._cov = self._montar_covariancia(matriz_correlacao)
        self.resultado = None
        self.sensibilidade_df = None
        self.precos_sombra = None
        self._pesos = None
        self._ativo_sens = None

    def _mascara_rf(self, df_temp: pd.DataFrame = None) -> np.ndarray:
        df_temp = self.df if df_temp is None else df_temp
        return (df_temp["Classe"] == self.CLASSE_RF).to_numpy()

    def _validar_limites(self):
        if (self.df["Teto_R$"] < self.df["Piso_R$"]).any():
            raise ValueError("Há ativos com teto individual menor que o piso.")
        if (self.df["Piso_R$"].sum() > self.capital_total + 1e-6
                or self.df["Teto_R$"].sum() < self.capital_total - 1e-6):
            raise ValueError("Os limites individuais inviabilizam alocar o capital total.")
        teto_rf = self.df.loc[self._mascara_rf(), "Teto_R$"].sum()
        if teto_rf < self.min_rf_pct * self.capital_total - 1e-6:
            raise ValueError("Os tetos da Renda Fixa não permitem cumprir o piso de Renda Fixa.")

    def _montar_covariancia(self, matriz_correlacao=None) -> np.ndarray:
        n = len(self.df)
        vol = self.df["Volatilidade_Anual"].to_numpy(dtype=float)

        if matriz_correlacao is not None:
            corr = np.asarray(matriz_correlacao, dtype=float)
            if corr.shape != (n, n):
                raise ValueError(f"matriz_correlacao deve ser {n}x{n}.")
        else:
            corr = np.full((n, n), self.corr_base)
            mm = (self.df["Classe"] == self.CLASSE_MM).to_numpy()
            corr[mm, :] = self.corr_multimercado
            corr[:, mm] = self.corr_multimercado
            np.fill_diagonal(corr, 1.0)

        return np.outer(vol, vol) * corr

    def _idx_ativo(self, nome_ativo: str) -> int:
        idx = self.df.index[self.df["Ativo"] == nome_ativo].tolist()
        if not idx:
            raise ValueError(f"Ativo '{nome_ativo}' não encontrado. Opções: {self.df['Ativo'].tolist()}")
        return idx[0]

    def _ativo_padrao_sensibilidade(self) -> str:
        mm = self.df.loc[self.df["Classe"] == self.CLASSE_MM, "Ativo"]
        return mm.iloc[0] if len(mm) else self.df["Ativo"].iloc[0]

    def _vol_carteira(self, pesos: np.ndarray) -> float:
        return float(np.sqrt(pesos @ self._cov @ pesos))

    def _montar_problema(self, df_temp: pd.DataFrame):
        n = len(df_temp)
        c = -df_temp["Retorno_Esperado"].to_numpy(dtype=float)
        A_eq = np.ones((1, n))
        b_eq = [self.capital_total]
        A_ub = -self._mascara_rf(df_temp).astype(float).reshape(1, -1)
        b_ub = [-self.min_rf_pct * self.capital_total]
        bounds = list(zip(df_temp["Piso_R$"], df_temp["Teto_R$"]))
        return c, A_ub, b_ub, A_eq, b_eq, bounds

    def _desempatar_por_risco(self, df_temp: pd.DataFrame, x_lp: np.ndarray) -> np.ndarray:
        cap = self.capital_total
        r = df_temp["Retorno_Esperado"].to_numpy(dtype=float)
        mask_rf = self._mascara_rf(df_temp)
        lb = df_temp["Piso_R$"].to_numpy(dtype=float) / cap
        ub = df_temp["Teto_R$"].to_numpy(dtype=float) / cap

        w0 = x_lp / cap
        r_opt = float(r @ w0)
        var0 = float(w0 @ self._cov @ w0)

        restricoes = LinearConstraint(
            np.vstack([np.ones_like(r), mask_rf.astype(float), r]),
            lb=[1.0, self.min_rf_pct, r_opt - self.TOL_RETORNO],
            ub=[1.0, np.inf, np.inf],
        )
        res = minimize(
            lambda w: 1e4 * (w @ self._cov @ w),
            w0,
            jac=lambda w: 2e4 * (self._cov @ w),
            hess=lambda w: 2e4 * self._cov,
            bounds=Bounds(lb, ub),
            constraints=[restricoes],
            method="trust-constr",
            options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 2000},
        )

        w = np.clip(res.x, lb, ub)
        valido = (
            r @ w >= r_opt - 1e-7
            and abs(w.sum() - 1.0) < 1e-7
            and w[mask_rf].sum() >= self.min_rf_pct - 1e-7
            and w @ self._cov @ w <= var0 + 1e-12
        )
        if not valido:
            warnings.warn("Desempate por risco não convergiu; mantendo a solução do LP.")
            return x_lp
        return w * cap

    def _resolver(self, df_temp: pd.DataFrame):
        c, A_ub, b_ub, A_eq, b_eq, bounds = self._montar_problema(df_temp)
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            return None, res
        x = res.x
        if self.minimizar_risco_no_empate:
            x = self._desempatar_por_risco(df_temp, x)
        return x, res

    def otimizar(self) -> pd.DataFrame:
        x, res = self._resolver(self.df)
        if x is None:
            raise RuntimeError(f"Falha na convergência: {res.message}")

        cap = self.capital_total
        self.resultado = res
        self._pesos = x / cap

        self.df["Alocacao_R$"] = np.round(x, 2)
        self.df["Alocacao_%"] = np.round(self._pesos * 100, 2)
        self.df["Retorno_Esperado_R$"] = np.round(x * self.df["Retorno_Esperado"], 2)
        self.df["Folga_Teto_R$"] = np.round(self.df["Teto_R$"] - x, 2)

        tetos = np.clip(-res.upper.marginals, 0.0, None)
        self.df["Preco_Sombra_Teto"] = np.round(tetos, 6)
        self.precos_sombra = {
            "capital": float(-res.eqlin.marginals[0]),
            "piso_rf": float(max(-res.ineqlin.marginals[0], 0.0)),
        }
        return self.df

    def calcular_metricas_risco(self, taxa_livre_risco: float = 0.1375) -> dict:
        if self.resultado is None:
            self.otimizar()

        pesos = self._pesos
        retorno = float(pesos @ self.df["Retorno_Esperado"].to_numpy(dtype=float))
        vol = self._vol_carteira(pesos)
        excesso = retorno - taxa_livre_risco
        sharpe = excesso / vol if vol > 1e-12 else np.nan

        return {
            "Retorno_Anual_%": retorno * 100,
            "Volatilidade_Anual_%": vol * 100,
            "Excesso_Retorno_%": excesso * 100,
            "Indice_Sharpe": sharpe,
            "Capital_Total": self.capital_total,
        }

    def simular_sensibilidade(self, nome_ativo: str = None, teto_min: float = 0.0,
                              teto_max: float = 0.60, passos: int = 25) -> pd.DataFrame:
        nome_ativo = nome_ativo or self._ativo_padrao_sensibilidade()
        idx_alvo = self._idx_ativo(nome_ativo)
        cap = self.capital_total
        r = self.df["Retorno_Esperado"].to_numpy(dtype=float)
        tetos_range = np.linspace(teto_min, teto_max, passos)
        retornos, vols = [], []

        for teto_val in tetos_range:
            df_temp = self.df.copy()
            df_temp.loc[idx_alvo, "Teto_R$"] = teto_val * cap
            try:
                x, _ = self._resolver(df_temp)
            except ValueError:
                x = None
            if x is None:
                retornos.append(np.nan)
                vols.append(np.nan)
            else:
                retornos.append(float(r @ x))
                vols.append(self._vol_carteira(x / cap) * 100)

        self._ativo_sens = nome_ativo
        self.sensibilidade_df = pd.DataFrame({
            "Teto_%": tetos_range * 100,
            "Retorno_R$": retornos,
            "Volatilidade_%": vols,
        })
        return self.sensibilidade_df

    def plotar_dashboard(self, salvar_como: str = None, ativo_sensibilidade: str = None,
                         mostrar: bool = True):
        if self.resultado is None:
            self.otimizar()
        ativo_sens = ativo_sensibilidade or self._ativo_padrao_sensibilidade()
        if self.sensibilidade_df is None or self._ativo_sens != ativo_sens:
            self.simular_sensibilidade(ativo_sens)

        cap = self.capital_total
        fig, axs = plt.subplots(2, 2, figsize=(15, 11))
        cores_ativos = [self.CORES_CLASSES.get(c, "#7f7f7f") for c in self.df["Classe"]]

        ax = axs[0, 0]
        bars = ax.barh(self.df["Ativo"], self.df["Alocacao_R$"], color=cores_ativos,
                       edgecolor="black", alpha=0.85)
        xmax = max(self.df["Teto_R$"].max(), self.df["Alocacao_R$"].max()) * 1.35
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        ax.set_title("Alocação Ótima por Ativo (R$)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Capital Investido (R$)")
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                        f"R$ {w:,.0f} ({w / cap * 100:.1f}%)",
                        va="center", fontsize=8.5, fontweight="bold")

        ax = axs[0, 1]
        por_classe = self.df.groupby("Classe")["Alocacao_R$"].sum()
        por_classe = por_classe[por_classe > 0]
        cores_classes = [self.CORES_CLASSES.get(c, "#7f7f7f") for c in por_classe.index]
        _, _, autotexts = ax.pie(
            por_classe, labels=por_classe.index, autopct="%1.1f%%", startangle=140,
            pctdistance=0.775, colors=cores_classes, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        ax.set_title("Exposição por Classe de Ativos", fontsize=11, fontweight="bold")

        ax = axs[1, 0]
        sens = self.sensibilidade_df
        l1, = ax.plot(sens["Teto_%"], sens["Retorno_R$"], marker="o", markersize=4,
                      color="#1b4f72", linewidth=2, label="Retorno (R$)")
        ax2 = ax.twinx()
        l2, = ax2.plot(sens["Teto_%"], sens["Volatilidade_%"], linestyle="--",
                       color="#b9770e", linewidth=1.8, label="Volatilidade (% a.a.)")

        idx_sens = self._idx_ativo(ativo_sens)
        teto_politica = self.df.loc[idx_sens, "Teto_Individual"] * 100
        handles = [l1, l2]
        handles.append(ax.axvline(teto_politica, color="crimson", linestyle="--", linewidth=1.5,
                                  label=f"Teto de Política ({teto_politica:.0f}%)"))
        limite_rf = (1 - self.min_rf_pct) * 100
        if sens["Teto_%"].min() <= limite_rf <= sens["Teto_%"].max():
            handles.append(ax.axvline(limite_rf, color="gray", linestyle=":", linewidth=1.5,
                                      label=f"Piso de RF passa a limitar ({limite_rf:.0f}%)"))
        ax.set_title(f"Trade-off de Política: Teto de {ativo_sens} vs. Retorno e Risco",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Teto de {ativo_sens} (%)")
        ax.set_ylabel("Retorno Anual (R$)")
        ax2.set_ylabel("Volatilidade da Carteira (% a.a.)")
        ax.legend(handles=handles, loc="lower right", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)

        ax = axs[1, 1]
        rotulos = [f"Piso Renda Fixa ({self.min_rf_pct:.0%})"]
        valores = [self.precos_sombra["piso_rf"] * 100]
        for nome, ps in zip(self.df["Ativo"], self.df["Preco_Sombra_Teto"]):
            rotulos.append(f"Teto {nome}")
            valores.append(ps * 100)
        ax.barh(rotulos, valores, color="#4a235a", edgecolor="black", alpha=0.85)
        ax.invert_yaxis()
        xmax = max(max(valores) * 1.45, 0.01)
        ax.set_xlim(0, xmax)
        ax.set_title("Preço-Sombra das Restrições", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ganho de retorno por R$ 1,00 de folga (% a.a.)")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        for y, v in enumerate(valores):
            ax.text(v + xmax * 0.01, y, f"{v:.3f}%" if v > 0 else "0", va="center",
                    fontsize=8.5, fontweight="bold")
        ax.text(0.98, 0.97,
                f"Preço-sombra do capital total: {self.precos_sombra['capital'] * 100:.2f}% a.a.",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, style="italic")

        plt.tight_layout()
        if salvar_como:
            plt.savefig(salvar_como, dpi=300)
        if mostrar:
            plt.show()
        return fig