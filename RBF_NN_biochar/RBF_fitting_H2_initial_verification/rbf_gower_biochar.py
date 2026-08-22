"""
rbf_gower_biochar.py
======================

Extension del aproximador RBF multicuadrico (desarrollado originalmente
solo para variables continuas, ver rbf_multiquadric.py / demo_rbf.py) para
soportar VARIABLES CATEGORICAS (nominales, sin orden) mezcladas con
variables continuas, mediante una distancia compuesta tipo Gower.

Aplicado aqui al dataset real de pirolisis catalitica de biochar,
modelando la fraccion molar de H2 en el gas de sintesis a partir de:

    Entradas continuas   : CARGA %, TR, N2, T
    Entradas categoricas : ORIGEN (3 niveles), CAT (3 niveles)

Por que una distancia especial para las categoricas
------------------------------------------------------
El nucleo multicuadrico phi(r) = sqrt(r^2+c^2) depende de una distancia
euclidiana r = ||x - x'||, que solo tiene sentido fisico cuando existe una
nocion de "cercania" entre valores (p.ej. 450 C esta mas cerca de 500 C que
de 800 C). Las variables ORIGEN y CAT son nominales: no existe un orden ni
una nocion de distancia natural entre, por ejemplo, "CaO" y "MgO". Codificar
esas categorias como enteros arbitrarios (CaO=1, MgO=2, No CAT=3) impondria
al modelo una estructura ordinal falsa.

La solucion adoptada es un analogo, para RBF, del COEFICIENTE DE
SIMILITUD DE GOWER (Gower, 1971), disenado precisamente para conjuntos de
datos con variables mixtas (continuas + categoricas + binarias). La idea de
Gower es promediar, sobre todas las variables, una disimilitud por variable:
para las continuas, una distancia normalizada; para las categoricas, un
"simple matching" que vale 0 si coinciden y 1 si no coinciden. Aqui se usa
la misma idea pero en la version "distancia al cuadrado" (para poder
insertarla directamente en el nucleo multicuadrico) y con un PESO AJUSTABLE
w_m por cada variable categorica, en vez de un promedio con pesos fijos:

    r^2(x,x') = ||x_cont - x'_cont||^2  +  sum_m  w_m * 1[cat_m != cat_m']

Los pesos w_m son hiperparametros adicionales, ajustados junto con el
parametro de forma c y la regularizacion lambda mediante Latin Hypercube
Sampling guiado por LOOCV (ver rbf_multiquadric.py para el desarrollo
completo de LOOCV analitico via la formula de Rippa/Allen, y la
justificacion de LHS sobre grid search).

Referencia principal para la distancia mixta:
    Gower, J.C. (1971). "A general coefficient of similarity and some of
    its properties". Biometrics, 27(4), 857-871.

Autor: adaptado para el dataset de pirolisis/biochar (ORIGEN, CAT, CARGA %,
TR, N2, T -> H2), Agosto 2026.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from scipy.stats import qmc


# ======================================================================
# 1. SOPORTE DE VARIABLES MIXTAS (continuas + categoricas)
# ======================================================================
class MixedFeatures:
    """
    Envuelve un bloque de variables continuas (ya estandarizadas) y uno o
    mas bloques categoricos (arreglos de strings), permitiendo indexado
    tipo numpy (X[indices]) y calculo de distancia mixta tipo Gower:

        r^2(x,x') = ||x_cont - x'_cont||^2 + sum_m w_m * 1[cat_m != cat_m']
    """

    def __init__(self, X_cont, cat_dict):
        self.X_cont = np.asarray(X_cont, dtype=float)
        self.cat_dict = {k: np.asarray(v) for k, v in cat_dict.items()}
        self.N = self.X_cont.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return MixedFeatures(self.X_cont[idx],
                              {k: v[idx] for k, v in self.cat_dict.items()})

    def sqdist(self, other, cat_weights):
        """Matriz (N, M) de distancias al cuadrado tipo Gower."""
        r2 = cdist(self.X_cont, other.X_cont, metric="sqeuclidean")
        for var, w in (cat_weights or {}).items():
            mismatch = (self.cat_dict[var][:, None] != other.cat_dict[var][None, :])
            r2 = r2 + w * mismatch.astype(float)
        return r2


# ======================================================================
# 2. NUCLEO RBF Y MATRIZ DE DISENO
# ======================================================================
def multiquadric(r, c):
    """phi(r) = sqrt(r^2 + c^2)"""
    return np.sqrt(r ** 2 + c ** 2)


def build_design_matrix(X, centers, c, poly=True, cat_weights=None, categories=None):
    """
    X, centers : instancias de MixedFeatures (variables continuas + categoricas).
    c          : shape parameter de la multicuadrica.
    poly       : si True, antepone columnas [1, x_cont] + dummies one-hot de
                 cada variable categorica (SIN penalizar), para capturar la
                 tendencia global y el efecto principal de cada categoria.
    cat_weights: dict {nombre_var_categorica: peso}.
    categories : dict {nombre_var_categorica: lista de categorias, orden fijo},
                 fijado en tiempo de entrenamiento para que el one-hot sea
                 consistente sin importar que subconjunto de filas se evalue.

    Devuelve Phi y n_unpenalized (num. de columnas no penalizadas).
    """
    r2 = X.sqdist(centers, cat_weights)
    Phi_rbf = np.sqrt(r2 + c ** 2)

    if poly:
        blocks = [np.ones((X.N, 1)), X.X_cont]
        for var, cats in (categories or {}).items():
            labels = X.cat_dict[var]
            onehot = (labels[:, None] == np.asarray(cats)[None, :]).astype(float)
            blocks.append(onehot)
        Phi_poly = np.hstack(blocks)
        Phi = np.hstack([Phi_poly, Phi_rbf])
        n_unpenalized = Phi_poly.shape[1]
    else:
        Phi = Phi_rbf
        n_unpenalized = 0

    return Phi, n_unpenalized


# ======================================================================
# 3. AJUSTE RIDGE + LOOCV ANALITICO (formula de Rippa/Allen, via Cholesky)
# ======================================================================
def ridge_fit_loocv(Phi, y, lam, n_unpenalized):
    """
    Ajusta w = argmin ||Phi w - y||^2 + lam*||w_penalizado||^2 y calcula el
    RMSE leave-one-out de forma EXACTA con un solo ajuste (formula de
    Rippa/Allen: e_i^LOO = e_i/(1-h_ii)), sin invertir matrices
    explicitamente (Cholesky + sustituciones triangulares -- ver discusion
    de estabilidad numerica en rbf_multiquadric.py).
    """
    n_cols = Phi.shape[1]
    P_diag = np.zeros(n_cols)
    P_diag[n_unpenalized:] = lam
    A = Phi.T @ Phi + np.diag(P_diag)

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
        w = A_inv @ (Phi.T @ y)
        h_diag = np.clip(np.diag(Phi @ A_inv @ Phi.T), 0, 1 - 1e-8)
        residual = y - Phi @ w
        rmse_train = np.sqrt(np.mean(residual ** 2))
        rmse_loocv = np.sqrt(np.mean((residual / (1 - h_diag)) ** 2))
        return w, rmse_train, rmse_loocv, h_diag

    b = Phi.T @ y
    z = solve_triangular(L, b, lower=True)
    w = solve_triangular(L.T, z, lower=False)

    V_T = solve_triangular(L, Phi.T, lower=True)
    h_diag = np.clip(np.sum(V_T ** 2, axis=0), 0, 1 - 1e-8)

    residual = y - Phi @ w
    loocv_residual = residual / (1.0 - h_diag)

    rmse_train = np.sqrt(np.mean(residual ** 2))
    rmse_loocv = np.sqrt(np.mean(loocv_residual ** 2))

    return w, rmse_train, rmse_loocv, h_diag


# ======================================================================
# 4. SELECCION GREEDY DE CENTROS GUIADA POR LOOCV
# ======================================================================
def greedy_select_centers(X, y, c, lam, max_centers=None, poly=True,
                           min_improve=1e-6, verbose=False,
                           cat_weights=None, categories=None):
    """
    Agrega, de entre los N puntos de entrenamiento, el candidato que mas
    reduce el RMSE-LOOCV en cada paso (Orthogonal-Least-Squares adaptado,
    Chen-Cowan-Grant 1991, guiado por LOOCV en vez de error de
    entrenamiento). Se detiene con early-stopping cuando ya no hay mejora
    significativa.
    """
    N = len(X)
    if max_centers is None:
        max_centers = N

    remaining = list(range(N))
    selected = []
    history_loocv = []
    best_loocv = np.inf

    while remaining and len(selected) < max_centers:
        trial_scores = []
        for cand in remaining:
            trial_set = selected + [cand]
            Phi, n_unpen = build_design_matrix(
                X, X[trial_set], c, poly=poly,
                cat_weights=cat_weights, categories=categories)
            _, _, rmse_loocv, _ = ridge_fit_loocv(Phi, y, lam, n_unpen)
            trial_scores.append(rmse_loocv)

        best_idx_local = int(np.argmin(trial_scores))
        best_candidate = remaining[best_idx_local]
        best_candidate_loocv = trial_scores[best_idx_local]

        improvement = best_loocv - best_candidate_loocv
        if len(selected) > 0 and improvement < min_improve:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_loocv = best_candidate_loocv
        history_loocv.append(best_loocv)

        if verbose:
            print(f"  centro {len(selected):2d}  ->  punto idx {best_candidate:3d}"
                  f"   RMSE-LOOCV = {best_loocv:.5f}")

    return selected, history_loocv


# ======================================================================
# 5. CLASE PRINCIPAL: RBFApproximator (con soporte de variables mixtas)
# ======================================================================
class RBFApproximator:
    """
    Aproximador RBF multicuadrico con seleccion greedy de centros (guiada
    por LOOCV analitico) y ajuste ridge, con soporte para variables de
    entrada continuas Y categoricas (distancia tipo Gower).

    La busqueda de hiperparametros (c, lambda, y un peso w_m por cada
    variable categorica) se hace mediante Latin Hypercube Sampling (LHS,
    McKay, Beckman & Conover, 1979) en dimension d=2+k, en vez de un grid
    regular -- ver Bergstra & Bengio (2012) para la justificacion general.

    Uso:
        modelo = RBFApproximator(cat_cols=['ORIGEN','CAT'],
                                  c_range=(0.01, 20.0), lam_range=(1e-8, 10.0),
                                  cat_weight_range=(0.001, 300.0),
                                  n_search_points=500, max_centers=40)
        modelo.fit(df[input_cols], df['H2'])
        y_pred = modelo.predict(df_new[input_cols])
    """

    def __init__(self, cat_cols=None, c_range=(0.1, 10.0), lam_range=(1e-6, 1.0),
                 cat_weight_range=(0.01, 100.0), n_search_points=195,
                 max_centers=None, poly=True, min_improve=1e-6,
                 search_seed=42, verbose=False):
        self.cat_cols = list(cat_cols) if cat_cols else []
        self.c_range = c_range
        self.lam_range = lam_range
        self.cat_weight_range = cat_weight_range
        self.n_search_points = n_search_points
        self.max_centers = max_centers
        self.poly = poly
        self.min_improve = min_improve
        self.search_seed = search_seed
        self.verbose = verbose

        self.cont_cols_ = None
        self.X_mean_ = None
        self.X_std_ = None
        self.categories_ = None
        self.centers_ = None
        self.w_ = None
        self.c_ = None
        self.lam_ = None
        self.cat_weights_ = None
        self.n_unpenalized_ = None
        self.rmse_loocv_ = None
        self.rmse_train_ = None
        self.history_loocv_ = None
        self.selected_idx_ = None
        self.search_history_ = None

    def _split_mixed(self, X, fit_mode):
        cont_cols = [c for c in X.columns if c not in self.cat_cols]
        if fit_mode:
            self.cont_cols_ = cont_cols

        X_cont = X[self.cont_cols_].to_numpy(dtype=float)
        if fit_mode:
            self.X_mean_ = X_cont.mean(axis=0)
            self.X_std_ = X_cont.std(axis=0)
            self.X_std_[self.X_std_ == 0] = 1.0
        X_cont_s = (X_cont - self.X_mean_) / self.X_std_

        cat_dict = {}
        for var in self.cat_cols:
            labels = X[var].to_numpy()
            if not fit_mode:
                unseen = set(np.unique(labels)) - set(self.categories_[var])
                if unseen:
                    raise ValueError(
                        f"Categoria(s) no vista(s) en entrenamiento para '{var}': "
                        f"{unseen}. El modelo no puede predecir para categorias "
                        f"desconocidas.")
            cat_dict[var] = labels

        return MixedFeatures(X_cont_s, cat_dict)

    def _sample_hyperparams(self):
        n_cat = len(self.cat_cols)
        d = 2 + n_cat
        bounds_log = [np.log10(self.c_range), np.log10(self.lam_range)]
        bounds_log += [np.log10(self.cat_weight_range)] * n_cat

        sampler = qmc.LatinHypercube(d=d, seed=self.search_seed)
        u = sampler.random(self.n_search_points)

        log_vals = np.array([
            lo + u[:, j] * (hi - lo) for j, (lo, hi) in enumerate(bounds_log)
        ]).T
        vals = 10 ** log_vals

        c_vals, lam_vals = vals[:, 0], vals[:, 1]
        cat_w_vals = {var: vals[:, 2 + i] for i, var in enumerate(self.cat_cols)}
        return c_vals, lam_vals, cat_w_vals

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame (se requieren "
                             "nombres de columna para separar continuas/categoricas).")
        self.categories_ = {var: sorted(X[var].unique()) for var in self.cat_cols}
        Xf = self._split_mixed(X, fit_mode=True)

        c_vals, lam_vals, cat_w_vals = self._sample_hyperparams()
        best = {"loocv": np.inf}
        self.search_history_ = []

        if self.verbose:
            print(f"Latin Hypercube Sampling: {self.n_search_points} puntos "
                  f"en dimension {2 + len(self.cat_cols)} "
                  f"(c, lambda, {', '.join('w_' + v for v in self.cat_cols)})")

        for i in range(self.n_search_points):
            c, lam = c_vals[i], lam_vals[i]
            cat_w = {var: cat_w_vals[var][i] for var in self.cat_cols}

            selected, hist = greedy_select_centers(
                Xf, y, c, lam,
                max_centers=self.max_centers,
                poly=self.poly,
                min_improve=self.min_improve,
                verbose=False,
                cat_weights=cat_w,
                categories=self.categories_,
            )
            if not hist:
                continue
            final_loocv = hist[-1]
            self.search_history_.append((c, lam, dict(cat_w), final_loocv))

            if final_loocv < best["loocv"]:
                best.update(loocv=final_loocv, c=c, lam=lam, cat_w=cat_w,
                            selected=selected, history=hist)

        if "selected" not in best:
            raise RuntimeError("No se pudo ajustar ningun modelo: revisa los datos.")

        self.c_ = best["c"]
        self.lam_ = best["lam"]
        self.cat_weights_ = best["cat_w"]
        self.selected_idx_ = best["selected"]
        self.history_loocv_ = best["history"]
        self.centers_ = Xf[self.selected_idx_]

        Phi, self.n_unpenalized_ = build_design_matrix(
            Xf, self.centers_, self.c_, poly=self.poly,
            cat_weights=self.cat_weights_, categories=self.categories_)
        self.w_, self.rmse_train_, self.rmse_loocv_, _ = ridge_fit_loocv(
            Phi, y, self.lam_, self.n_unpenalized_)

        if self.verbose:
            cat_w_str = ", ".join(f"w_{v}={self.cat_weights_[v]:.3g}" for v in self.cat_cols)
            print(f"\nMejor modelo: c={self.c_:.4g}  lambda={self.lam_:.2e}, {cat_w_str}"
                  f"  n_centros={len(self.selected_idx_)}")
            print(f"RMSE entrenamiento = {self.rmse_train_:.5f}")
            print(f"RMSE LOOCV         = {self.rmse_loocv_:.5f}")

        return self

    def predict(self, X_new):
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("X_new debe ser un pandas DataFrame.")
        Xf_new = self._split_mixed(X_new, fit_mode=False)
        Phi, _ = build_design_matrix(Xf_new, self.centers_, self.c_, poly=self.poly,
                                      cat_weights=self.cat_weights_, categories=self.categories_)
        return Phi @ self.w_

    def summary(self):
        print("=" * 55)
        print("RESUMEN DEL MODELO RBF (Gower, mixto) -- H2")
        print("=" * 55)
        n_dim = 2 + len(self.cat_cols)
        print(f"Metodo de busqueda        : Latin Hypercube Sampling "
              f"({self.n_search_points} evaluaciones, dim={n_dim})")
        print(f"Shape parameter c         : {self.c_:.4g}")
        print(f"Regularizacion lambda     : {self.lam_:.2e}")
        for var in self.cat_cols:
            print(f"Peso categorico w_{var:<10}: {self.cat_weights_[var]:.4g}")
        print(f"Numero de centros usados  : {len(self.selected_idx_)}")
        print(f"RMSE entrenamiento        : {self.rmse_train_:.5f}")
        print(f"RMSE LOOCV (generalizacion): {self.rmse_loocv_:.5f}")
        print("=" * 55)


# ======================================================================
# 6. SCRIPT PRINCIPAL: ajuste de H2 sobre el dataset real de biochar
# ======================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 6.1 Carga de datos (desde el Excel ya reorganizado con entradas/salidas
    #     separadas y documentadas -- ver informe tecnico adjunto)
    # ------------------------------------------------------------------
    RUTA_EXCEL = "dataset_biochar_reorganizado.xlsx"
    df = pd.read_excel(RUTA_EXCEL, sheet_name="Datos", header=4)
    df = df.rename(columns={"T (°C)": "T"})

    cat_cols = ["ORIGEN", "CAT"]
    cont_cols = ["CARGA %", "TR", "N2", "T"]
    input_cols = cat_cols + cont_cols
    target = "H2"

    X = df[input_cols].copy()
    y = df[target].values

    # ------------------------------------------------------------------
    # 6.2 Ajuste del modelo (rangos finales, validados por no quedar
    #     pegados a los bordes tras varias corridas de refinamiento --
    #     ver informe tecnico, Seccion 5)
    # ------------------------------------------------------------------
    modelo = RBFApproximator(
        cat_cols=cat_cols,
        c_range=(0.01, 20.0),
        lam_range=(1e-8, 10.0),
        cat_weight_range=(0.001, 300.0),
        n_search_points=500,
        max_centers=40,
        search_seed=123,
        verbose=True,
    )
    modelo.fit(X, y)
    modelo.summary()

    y_hat = modelo.predict(X)
    residuos = y - y_hat
    r2 = 1 - np.sum(residuos ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"R^2 (entrenamiento) = {r2:.4f}")

    # ------------------------------------------------------------------
    # 6.3 Diagnosticos: convergencia, predicho vs. real, residuos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(range(1, len(modelo.history_loocv_) + 1), modelo.history_loocv_,
            marker="o", ms=3)
    ax.set_xlabel("Numero de centros seleccionados")
    ax.set_ylabel("RMSE-LOOCV")
    ax.set_title("Convergencia de la seleccion greedy (H2)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y, y_hat, alpha=0.7, edgecolor="k", linewidth=0.3)
    lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("H2 real")
    ax.set_ylabel("H2 predicho")
    ax.set_title(f"Predicho vs. real (R2={r2:.3f})")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(y_hat, residuos, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("H2 predicho")
    ax.set_ylabel("Residuo")
    ax.set_title("Residuos vs. predicho")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(residuos, bins=15, edgecolor="k", alpha=0.75)
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion de residuos")
    ax.grid(alpha=0.3)

    fig.suptitle("Diagnosticos del modelo RBF para H2 (datos reales biochar)", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig("diagnosticos_H2.png", dpi=150, bbox_inches="tight")
    print("Guardado: diagnosticos_H2.png")

    # ------------------------------------------------------------------
    # 6.4 Dependencia parcial: variables continuas, estratificadas por CAT
    #     (w_CAT domina la distancia categorica, asi que interesa ver si
    #     el efecto de cada variable continua cambia segun el catalizador)
    # ------------------------------------------------------------------
    cat_levels = {v: sorted(df[v].unique()) for v in cat_cols}
    origen_fijo = df["ORIGEN"].mode()[0]
    means = df[cont_cols].mean()
    colors = {"CaO": "#1f77b4", "MgO": "#ff7f0e", "No CAT": "#2ca02c"}

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for j, var in enumerate(cont_cols):
        ax = axes2[j]
        xr = np.linspace(df[var].min(), df[var].max(), 60)
        for cat in cat_levels["CAT"]:
            rows = []
            for xv in xr:
                row = means.to_dict()
                row[var] = xv
                row["ORIGEN"] = origen_fijo
                row["CAT"] = cat
                rows.append(row)
            Xg = pd.DataFrame(rows)[input_cols]
            yg = modelo.predict(Xg)
            ax.plot(xr, yg, label=cat, color=colors[cat], lw=2)
        ax.scatter(df[var], df[target], s=10, alpha=0.25, color="gray")
        ax.set_xlabel(var)
        if j == 0:
            ax.set_ylabel(f"{target} (predicho)")
        ax.set_title(f"Efecto de {var}\n(ORIGEN={origen_fijo}, otras en su media)")
        ax.grid(alpha=0.3)
    axes2[0].legend(title="CAT", fontsize=8)
    fig2.tight_layout()
    fig2.savefig("dependencia_parcial_H2_continuas.png", dpi=150, bbox_inches="tight")
    print("Guardado: dependencia_parcial_H2_continuas.png")

    # ------------------------------------------------------------------
    # 6.5 Efecto principal de las variables categoricas
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, var in zip(axes3, cat_cols):
        preds = []
        for level in cat_levels[var]:
            row = means.to_dict()
            row["ORIGEN"] = origen_fijo
            row["CAT"] = cat_levels["CAT"][0]
            row[var] = level
            Xg = pd.DataFrame([row])[input_cols]
            preds.append(modelo.predict(Xg)[0])
        ax.bar(cat_levels[var], preds, color="#4c72b0", edgecolor="k")
        ax.set_title(f"Efecto principal de {var}\n(resto en referencia/media)")
        ax.set_ylabel(f"{target} (predicho)")
        ax.grid(alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig("dependencia_parcial_H2_categoricas.png", dpi=150, bbox_inches="tight")
    print("Guardado: dependencia_parcial_H2_categoricas.png")
