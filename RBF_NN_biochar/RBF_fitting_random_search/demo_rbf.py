"""
demo_rbf.py
============

Ejemplo de uso de rbf_multiquadric.RBFApproximator sobre datos
experimentales z = f(x1, x2, x3, x4, x5).

CON TUS DATOS REALES:
----------------------
Sustituye el bloque "CARGA DE DATOS" por, por ejemplo:

    import pandas as pd
    df = pd.read_csv("tus_datos.csv")   # columnas: x1,x2,x3,x4,x5,z
    X = df[["x1", "x2", "x3", "x4", "x5"]].to_numpy()
    y = df["z"].to_numpy()

y sigue igual desde ahi.
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_multiquadric import RBFApproximator


# ------------------------------------------------------------------
# CARGA DE DATOS (aqui: sintetico, con ruido, simulando tu ensayo)
# ------------------------------------------------------------------
np.random.seed(0)
N, d = 84, 5
X = np.random.uniform(-2, 2, size=(N, d))
z_true = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] * X[:, 3]
          + 0.3 * X[:, 4] + 0.2 * np.cos(X[:, 0] * X[:, 4]))
ruido = np.random.normal(0, 0.15 * np.std(z_true), size=N)
z = z_true + ruido

var_names = [f"x{i+1}" for i in range(d)]

# ------------------------------------------------------------------
# AJUSTE DEL MODELO
# ------------------------------------------------------------------
# Busqueda de hiperparametros mediante Latin Hypercube Sampling (en vez de
# grid search), con el mismo presupuesto/rango que la version anterior
# (n_search_points=195 equivale al grid 15x13 usado originalmente).
# Con N=84 y max_centers=40 el ajuste completo tarda ~1 min.
modelo = RBFApproximator(
    c_range=(0.1, 10.0),
    lam_range=(1e-6, 1.0),
    n_search_points=195,
    max_centers=40,
    verbose=True,
)
modelo.fit(X, z)
modelo.summary()

y_hat = modelo.predict(X)
residuos = z - y_hat
r2 = 1 - np.sum(residuos ** 2) / np.sum((z - z.mean()) ** 2)
print(f"R^2 (entrenamiento)      : {r2:.4f}")

# ------------------------------------------------------------------
# DIAGNOSTICOS GRAFICOS
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# (1) Curva de convergencia del greedy (RMSE-LOOCV vs numero de centros)
ax = axes[0, 0]
ax.plot(range(1, len(modelo.history_loocv_) + 1), modelo.history_loocv_,
        marker="o", ms=3)
ax.set_xlabel("Numero de centros seleccionados")
ax.set_ylabel("RMSE - LOOCV")
ax.set_title("Convergencia de la seleccion greedy de centros")
ax.grid(alpha=0.3)

# (2) Predicho vs. real
ax = axes[0, 1]
ax.scatter(z, y_hat, alpha=0.7, edgecolor="k", linewidth=0.3)
lims = [min(z.min(), y_hat.min()), max(z.max(), y_hat.max())]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel("z real")
ax.set_ylabel("z predicho")
ax.set_title(f"Predicho vs. real  (R2={r2:.3f})")
ax.grid(alpha=0.3)

# (3) Residuos vs. predicho
ax = axes[1, 0]
ax.scatter(y_hat, residuos, alpha=0.7, edgecolor="k", linewidth=0.3)
ax.axhline(0, color="r", ls="--", lw=1)
ax.set_xlabel("z predicho")
ax.set_ylabel("Residuo (z - z_hat)")
ax.set_title("Residuos vs. predicho")
ax.grid(alpha=0.3)

# (4) Histograma de residuos
ax = axes[1, 1]
ax.hist(residuos, bins=15, edgecolor="k", alpha=0.75)
ax.set_xlabel("Residuo")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribucion de residuos")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("diagnosticos_rbf.png", dpi=150)
print("Guardado: diagnosticos_rbf.png")


# ------------------------------------------------------------------
# CURVAS DE DEPENDENCIA PARCIAL (una por variable, resto fijo en su media)
# ------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, d, figsize=(4 * d, 3.5), sharey=True)
x_mean = X.mean(axis=0)

for j in range(d):
    xj_range = np.linspace(X[:, j].min(), X[:, j].max(), 100)
    X_grid = np.tile(x_mean, (100, 1))
    X_grid[:, j] = xj_range
    y_grid = modelo.predict(X_grid)

    ax = axes2[j]
    ax.plot(xj_range, y_grid, lw=2)
    ax.scatter(X[:, j], z, s=12, alpha=0.35, color="gray")
    ax.set_xlabel(var_names[j])
    if j == 0:
        ax.set_ylabel("z (predicho, otras var. en su media)")
    ax.set_title(f"Efecto de {var_names[j]}")
    ax.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig("dependencia_parcial_rbf.png", dpi=150)
print("Guardado: dependencia_parcial_rbf.png")

plt.close("all")
