"""
rbf_multiquadric.py
====================

Aproximacion de datos experimentales dispersos y ruidosos mediante una red
RBF (Radial Basis Function) con nucleo multicuadrico:

        phi(r) = sqrt(r^2 + c^2)

Disenado para el caso: pocos datos (N ~ 84), varias variables independientes
(d = 5), y ruido experimental no despreciable.

Estrategia (resumen):
----------------------
1. Estandarizacion de las variables independientes (z-score), indispensable
   para que la distancia euclidiana r tenga sentido cuando las variables
   tienen escalas distintas.

2. Modelo:   z_hat(x) = [1, x_1,...,x_d] . beta   +   sum_j w_j * phi(||x-c_j||)

   Se incluye una cola polinomica (constante + lineal) SIN penalizar, que
   captura la tendencia global y mejora el comportamiento en extrapolacion
   (practica estandar para RBFs condicionalmente definidas positivas como
   la multicuadrica).

3. Ajuste de los pesos w (y beta) por MINIMOS CUADRADOS REGULARIZADOS
   (ridge / Tikhonov) sobre los pesos RBF unicamente. Esto evita la
   interpolacion exacta (que en datos ruidosos es sobreajuste garantizado).

4. SELECCION GREEDY DE CENTROS (Orthogonal-Least-Squares / Error-Reduction
   Ratio, Chen-Cowan-Grant 1991, adaptado): en cada iteracion se prueba
   agregar, de entre los puntos de entrenamiento aun no usados como centro,
   aquel que MAS REDUCE EL ERROR LOOCV (leave-one-out) analitico -- no el
   error de entrenamiento. El proceso se detiene cuando el LOOCV deja de
   mejorar (early stopping), evitando que el algoritmo "memorice" ruido.

5. El LOOCV se calcula de forma ANALITICA (formula de Rippa, estandar en
   RBF/ridge regression) usando la matriz "hat" H = Phi (Phi^T Phi + P)^-1
   Phi^T, sin tener que reajustar el modelo N veces. Esto hace que la
   busqueda greedy + grid search de hiperparametros sea barata
   computacionalmente incluso repitiendola muchas veces.

6. GRID SEARCH sobre el parametro de forma c y el parametro de
   regularizacion lambda, eligiendo la combinacion que produce el menor
   RMSE-LOOCV tras la seleccion greedy de centros.

Autor: (adaptado para tu ensayo RBF, 5 variables independientes, N=84)
"""

import numpy as np
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------
# Nucleo RBF
# ----------------------------------------------------------------------
def multiquadric(r, c):
    """phi(r) = sqrt(r^2 + c^2)"""
    return np.sqrt(r ** 2 + c ** 2)


# ----------------------------------------------------------------------
# Construccion de la matriz de diseno (con cola polinomica opcional)
# ----------------------------------------------------------------------
def build_design_matrix(X, centers, c, poly=True):
    """
    X       : (N, d)  puntos donde se evalua el modelo
    centers : (M, d)  centros RBF
    c       : shape parameter de la multicuadrica
    poly    : si True, antepone columnas [1, x_1,...,x_d] (sin penalizar)

    Devuelve Phi de tamano (N, M) o (N, M + d + 1) si poly=True,
    y n_unpenalized = numero de columnas polinomicas (0 si poly=False).
    """
    R = cdist(X, centers, metric="euclidean")
    Phi_rbf = multiquadric(R, c)

    if poly:
        N = X.shape[0]
        Phi_poly = np.hstack([np.ones((N, 1)), X])
        Phi = np.hstack([Phi_poly, Phi_rbf])
        n_unpenalized = Phi_poly.shape[1]
    else:
        Phi = Phi_rbf
        n_unpenalized = 0

    return Phi, n_unpenalized


# ----------------------------------------------------------------------
# Ajuste ridge + LOOCV analitico (formula de Rippa)
# ----------------------------------------------------------------------
def ridge_fit_loocv(Phi, y, lam, n_unpenalized):
    """
    Ajusta  w = argmin ||Phi w - y||^2 + lam * ||w_penalizado||^2
    (las columnas polinomicas, indices [0, n_unpenalized), NO se penalizan)

    Devuelve:
        w          : vector de pesos
        rmse_train : RMSE sobre los datos de entrenamiento
        rmse_loocv : RMSE leave-one-out (analitico, sin refit)
        h_diag     : diagonal de la matriz "hat" (leverage de cada punto)
    """
    n_cols = Phi.shape[1]
    P = np.eye(n_cols) * lam
    P[:n_unpenalized, :n_unpenalized] = 0.0  # no penalizar la cola polinomica

    A = Phi.T @ Phi + P
    # Se usa lstsq via solve con pequena regularizacion numerica adicional
    # solo si A esta mal condicionada (evita errores por matrices casi
    # singulares cuando hay pocos centros o c muy grande/pequeno).
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    w = A_inv @ (Phi.T @ y)
    y_hat = Phi @ w
    residual = y - y_hat

    H = Phi @ A_inv @ Phi.T
    h_diag = np.clip(np.diag(H), 0, 1 - 1e-8)  # evitar division por 0

    loocv_residual = residual / (1.0 - h_diag)

    rmse_train = np.sqrt(np.mean(residual ** 2))
    rmse_loocv = np.sqrt(np.mean(loocv_residual ** 2))

    return w, rmse_train, rmse_loocv, h_diag


# ----------------------------------------------------------------------
# Seleccion GREEDY de centros guiada por LOOCV
# ----------------------------------------------------------------------
def greedy_select_centers(X, y, c, lam, max_centers=None, poly=True,
                           min_improve=1e-6, verbose=False):
    """
    Selecciona, de entre los N puntos de entrenamiento, un subconjunto de
    centros RBF de forma greedy: en cada paso se agrega el candidato que
    mas reduce el RMSE-LOOCV. Se detiene cuando ya no hay mejora
    significativa (min_improve) o se alcanza max_centers.

    Devuelve:
        selected      : lista de indices (en X) elegidos como centros
        history_loocv : RMSE-LOOCV tras cada paso greedy
    """
    N = X.shape[0]
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
            Phi, n_unpen = build_design_matrix(X, X[trial_set], c, poly=poly)
            _, _, rmse_loocv, _ = ridge_fit_loocv(Phi, y, lam, n_unpen)
            trial_scores.append(rmse_loocv)

        best_idx_local = int(np.argmin(trial_scores))
        best_candidate = remaining[best_idx_local]
        best_candidate_loocv = trial_scores[best_idx_local]

        improvement = best_loocv - best_candidate_loocv
        if len(selected) > 0 and improvement < min_improve:
            break  # ya no mejora: detener seleccion (early stopping)

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_loocv = best_candidate_loocv
        history_loocv.append(best_loocv)

        if verbose:
            print(f"  centro {len(selected):2d}  ->  punto idx {best_candidate:3d}"
                  f"   RMSE-LOOCV = {best_loocv:.5f}")

    return selected, history_loocv


# ----------------------------------------------------------------------
# Clase principal: API tipo sklearn (fit / predict)
# ----------------------------------------------------------------------
class RBFApproximator:
    """
    Aproximador RBF multicuadrico con seleccion greedy de centros
    (guiada por LOOCV) y ajuste ridge, para datos experimentales
    dispersos y ruidosos.

    Uso:
        modelo = RBFApproximator(c_grid=..., lam_grid=..., max_centers=...)
        modelo.fit(X, y)
        y_pred = modelo.predict(X_new)
    """

    def __init__(self, c_grid=None, lam_grid=None, max_centers=None,
                 poly=True, min_improve=1e-6, verbose=False):
        self.c_grid = c_grid if c_grid is not None else np.logspace(-1, 1, 15)
        self.lam_grid = lam_grid if lam_grid is not None else np.logspace(-6, 0, 13)
        self.max_centers = max_centers
        self.poly = poly
        self.min_improve = min_improve
        self.verbose = verbose

        # se llenan en fit()
        self.X_mean_ = None
        self.X_std_ = None
        self.centers_ = None
        self.w_ = None
        self.c_ = None
        self.lam_ = None
        self.n_unpenalized_ = None
        self.rmse_loocv_ = None
        self.rmse_train_ = None
        self.history_loocv_ = None
        self.selected_idx_ = None

    # ---- estandarizacion -------------------------------------------------
    def _standardize(self, X):
        return (X - self.X_mean_) / self.X_std_

    # ---- entrenamiento -----------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_std_[self.X_std_ == 0] = 1.0  # evitar division por cero
        Xs = self._standardize(X)

        best = {"loocv": np.inf}

        if self.verbose:
            print(f"Grid search: {len(self.c_grid)} valores de c x "
                  f"{len(self.lam_grid)} valores de lambda")

        for c in self.c_grid:
            for lam in self.lam_grid:
                selected, hist = greedy_select_centers(
                    Xs, y, c, lam,
                    max_centers=self.max_centers,
                    poly=self.poly,
                    min_improve=self.min_improve,
                    verbose=False,
                )
                if not hist:
                    continue
                final_loocv = hist[-1]

                if final_loocv < best["loocv"]:
                    best.update(loocv=final_loocv, c=c, lam=lam,
                                selected=selected, history=hist)

        if "selected" not in best:
            raise RuntimeError("No se pudo ajustar ningun modelo: revisa los datos.")

        self.c_ = best["c"]
        self.lam_ = best["lam"]
        self.selected_idx_ = best["selected"]
        self.history_loocv_ = best["history"]
        self.centers_ = Xs[self.selected_idx_]

        Phi, self.n_unpenalized_ = build_design_matrix(
            Xs, self.centers_, self.c_, poly=self.poly)
        self.w_, self.rmse_train_, self.rmse_loocv_, _ = ridge_fit_loocv(
            Phi, y, self.lam_, self.n_unpenalized_)

        if self.verbose:
            print(f"\nMejor modelo: c={self.c_:.4g}  lambda={self.lam_:.2e}  "
                  f"n_centros={len(self.selected_idx_)}")
            print(f"RMSE entrenamiento = {self.rmse_train_:.5f}")
            print(f"RMSE LOOCV         = {self.rmse_loocv_:.5f}")

        return self

    # ---- prediccion --------------------------------------------------
    def predict(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        Xs_new = self._standardize(X_new)
        Phi, _ = build_design_matrix(Xs_new, self.centers_, self.c_, poly=self.poly)
        return Phi @ self.w_

    # ---- resumen -------------------------------------------------------
    def summary(self):
        print("=" * 55)
        print("RESUMEN DEL MODELO RBF MULTICUADRICO")
        print("=" * 55)
        print(f"Shape parameter c        : {self.c_:.4g}")
        print(f"Regularizacion lambda     : {self.lam_:.2e}")
        print(f"Numero de centros usados  : {len(self.selected_idx_)} "
              f"(de {self.X_mean_.shape[0] and len(self.centers_)} candidatos evaluados)")
        print(f"RMSE entrenamiento        : {self.rmse_train_:.5f}")
        print(f"RMSE LOOCV (generalizacion): {self.rmse_loocv_:.5f}")
        print("=" * 55)
