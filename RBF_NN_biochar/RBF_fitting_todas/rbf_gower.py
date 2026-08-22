"""
rbf_gower.py
=============

Modulo compartido: aproximador RBF multicuadrico con distancia mixta tipo
Gower (para variables categoricas + continuas), seleccion greedy de centros
guiada por LOOCV analitico (formula de Rippa/Allen via Cholesky), y busqueda
de hiperparametros mediante Latin Hypercube Sampling.

Ver el informe tecnico "informe_biochar_H2.pdf" para la derivacion completa
y las referencias (Gower 1971; Rippa 1999; Chen-Cowan-Grant 1991; Allen 1974;
McKay-Beckman-Conover 1979; Bergstra-Bengio 2012).

Este modulo NO ajusta ningun modelo por si solo: cada salida (H2, CH4, CO2,
CENIZA, % Solido) tiene su propio script "fit_<salida>.py" que importa de
aqui, porque cada salida es un problema de aproximacion independiente con
sus propios hiperparametros optimos -- no deben asumirse compartidos entre
salidas.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from scipy.stats import qmc


# ======================================================================
# Soporte de variables MIXTAS (continuas + categoricas)
# ======================================================================
class MixedFeatures:
    """
    Envuelve un bloque de variables continuas (ya estandarizadas) y uno o
    mas bloques categoricos (arreglos de strings), con distancia mixta
    tipo Gower:  r^2(x,x') = ||x_cont-x'_cont||^2 + sum_m w_m*1[cat_m!=cat_m']
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
        r2 = cdist(self.X_cont, other.X_cont, metric="sqeuclidean")
        for var, w in (cat_weights or {}).items():
            mismatch = (self.cat_dict[var][:, None] != other.cat_dict[var][None, :])
            r2 = r2 + w * mismatch.astype(float)
        return r2


# ======================================================================
# Nucleo RBF y matriz de diseno
# ======================================================================
def multiquadric(r, c):
    """phi(r) = sqrt(r^2 + c^2)"""
    return np.sqrt(r ** 2 + c ** 2)


def build_design_matrix(X, centers, c, poly=True, cat_weights=None, categories=None):
    """
    X, centers : instancias de MixedFeatures.
    Devuelve Phi y n_unpenalized (num. de columnas no penalizadas: constante
    + lineales continuas + dummies one-hot de cada categorica).
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
# Ajuste ridge + LOOCV analitico (formula de Rippa/Allen, via Cholesky)
# ======================================================================
def ridge_fit_loocv(Phi, y, lam, n_unpenalized):
    """
    w = argmin ||Phi w - y||^2 + lam*||w_penalizado||^2, y RMSE-LOOCV exacto
    con un solo ajuste (e_i^LOO = e_i/(1-h_ii)), sin invertir matrices
    explicitamente (Cholesky + sustituciones triangulares).
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
# Seleccion greedy de centros guiada por LOOCV
# ======================================================================
def greedy_select_centers(X, y, c, lam, max_centers=None, poly=True,
                           min_improve=1e-6, verbose=False,
                           cat_weights=None, categories=None):
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
# Clase principal: RBFApproximator (soporte de variables mixtas)
# ======================================================================
class RBFApproximator:
    """
    Aproximador RBF multicuadrico con seleccion greedy de centros (LOOCV
    analitico) y busqueda de hiperparametros (c, lambda, y un peso w_m por
    variable categorica) mediante Latin Hypercube Sampling.

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
                        f"{unseen}.")
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
            raise TypeError("X debe ser un pandas DataFrame.")
        self.categories_ = {var: sorted(X[var].unique()) for var in self.cat_cols}
        Xf = self._split_mixed(X, fit_mode=True)

        c_vals, lam_vals, cat_w_vals = self._sample_hyperparams()
        best = {"loocv": np.inf}
        self.search_history_ = []

        if self.verbose:
            print(f"Latin Hypercube Sampling: {self.n_search_points} puntos "
                  f"en dimension {2 + len(self.cat_cols)}")

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

    def summary(self, nombre_salida=""):
        print("=" * 55)
        print(f"RESUMEN DEL MODELO RBF (Gower, mixto) -- {nombre_salida}")
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

    def frac_en_rango(self):
        """
        Diagnostico de 'pinning': posicion relativa (0=borde inferior,
        1=borde superior) de cada hiperparametro final dentro de su rango
        de busqueda, en escala log. Valores <0.05 o >0.95 son señal de
        alerta: el rango deberia ampliarse en esa direccion y reajustar.
        """
        def frac(val, lo, hi):
            return (np.log10(val) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))

        out = {
            "c": frac(self.c_, *self.c_range),
            "lambda": frac(self.lam_, *self.lam_range),
        }
        for var in self.cat_cols:
            out[f"w_{var}"] = frac(self.cat_weights_[var], *self.cat_weight_range)
        return out
