"""
fit_CENIZA.py
==========
Ajusta el modelo RBF (Gower, mixto) para CENIZA sobre el ensayo de
pirolisis catalitica de biochar. Rangos e hiperparametros validados por
busqueda de multiples semillas LHS (ver bitacora en el chat / informe
tecnico); NO se asumieron iguales a los de otras salidas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rbf_gower import RBFApproximator

RUTA_EXCEL = "dataset_biochar_reorganizado.xlsx"
TARGET = "CENIZA"
TARGET_SLUG = "CENIZA"
CAT_COLS = ["ORIGEN", "CAT"]
CONT_COLS = ["CARGA %", "TR", "N2", "T"]
INPUT_COLS = CAT_COLS + CONT_COLS

# ----------------------------------------------------------------------
# 1. Carga de datos
# ----------------------------------------------------------------------
df = pd.read_excel(RUTA_EXCEL, sheet_name="Datos", header=4)
df = df.rename(columns={"T (°C)": "T"})
X = df[INPUT_COLS].copy()
y = df[TARGET].values

# ----------------------------------------------------------------------
# 2. Ajuste (rangos y semilla validados especificamente para CENIZA)
# ----------------------------------------------------------------------
modelo = RBFApproximator(
    cat_cols=CAT_COLS,
    c_range=(0.01, 20.0),
    lam_range=(1e-10, 10.0),
    cat_weight_range=(0.0001, 3000.0),
    n_search_points=250,
    max_centers=40,
    search_seed=9,
    verbose=True,
)
modelo.fit(X, y)
modelo.summary(TARGET)
print("Posicion en rango (0=borde inf., 1=borde sup.):",
      {k: round(v, 3) for k, v in modelo.frac_en_rango().items()})

y_hat = modelo.predict(X)
residuos = y - y_hat
r2 = 1 - np.sum(residuos ** 2) / np.sum((y - y.mean()) ** 2)
r2_loocv = 1 - (modelo.rmse_loocv_ / y.std()) ** 2
print(f"R^2 (entrenamiento) = {r2:.4f}   R^2 (LOOCV, aprox.) = {r2_loocv:.4f}")

# ----------------------------------------------------------------------
# 3. Diagnosticos
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
ax.plot(range(1, len(modelo.history_loocv_) + 1), modelo.history_loocv_, marker="o", ms=3)
ax.set_xlabel("Numero de centros seleccionados")
ax.set_ylabel("RMSE-LOOCV")
ax.set_title(f"Convergencia de la seleccion greedy ({TARGET})")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(y, y_hat, alpha=0.7, edgecolor="k", linewidth=0.3)
lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel(f"{TARGET} real")
ax.set_ylabel(f"{TARGET} predicho")
ax.set_title(f"Predicho vs. real (R2={r2:.3f})")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(y_hat, residuos, alpha=0.7, edgecolor="k", linewidth=0.3)
ax.axhline(0, color="r", ls="--", lw=1)
ax.set_xlabel(f"{TARGET} predicho")
ax.set_ylabel("Residuo")
ax.set_title("Residuos vs. predicho")
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.hist(residuos, bins=15, edgecolor="k", alpha=0.75)
ax.set_xlabel("Residuo")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribucion de residuos")
ax.grid(alpha=0.3)

fig.suptitle(f"Diagnosticos del modelo RBF para {TARGET} (datos reales biochar)", y=1.00, fontsize=13)
fig.tight_layout()
fig.savefig(f"diagnosticos_{TARGET_SLUG}.png", dpi=150, bbox_inches="tight")
print(f"Guardado: diagnosticos_{TARGET_SLUG}.png")

# ----------------------------------------------------------------------
# 4. Dependencia parcial: continuas, estratificadas por la categorica
#    dominante para ESTA salida (la de mayor peso w_m, no asumida fija)
# ----------------------------------------------------------------------
cat_dominante = max(CAT_COLS, key=lambda v: modelo.cat_weights_[v])
cat_secundaria = [v for v in CAT_COLS if v != cat_dominante][0]
print(f"Categorica dominante para {TARGET}: {cat_dominante} "
      f"(w={modelo.cat_weights_[cat_dominante]:.4g} vs. "
      f"w_{cat_secundaria}={modelo.cat_weights_[cat_secundaria]:.4g})")

cat_levels = {v: sorted(df[v].unique()) for v in CAT_COLS}
nivel_fijo_secundaria = df[cat_secundaria].mode()[0]
means = df[CONT_COLS].mean()
palette = plt.cm.tab10.colors

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
for j, var in enumerate(CONT_COLS):
    ax = axes2[j]
    xr = np.linspace(df[var].min(), df[var].max(), 60)
    for k, level in enumerate(cat_levels[cat_dominante]):
        rows = []
        for xv in xr:
            row = means.to_dict()
            row[var] = xv
            row[cat_secundaria] = nivel_fijo_secundaria
            row[cat_dominante] = level
            rows.append(row)
        Xg = pd.DataFrame(rows)[INPUT_COLS]
        yg = modelo.predict(Xg)
        ax.plot(xr, yg, label=str(level), color=palette[k % 10], lw=2)
    ax.scatter(df[var], df[TARGET], s=10, alpha=0.25, color="gray")
    ax.set_xlabel(var)
    if j == 0:
        ax.set_ylabel(f"{TARGET} (predicho)")
    ax.set_title(f"Efecto de {var}\n({cat_secundaria}={nivel_fijo_secundaria}, otras en su media)")
    ax.grid(alpha=0.3)
axes2[0].legend(title=cat_dominante, fontsize=8)
fig2.tight_layout()
fig2.savefig(f"dependencia_parcial_{TARGET_SLUG}_continuas.png", dpi=150, bbox_inches="tight")
print(f"Guardado: dependencia_parcial_{TARGET_SLUG}_continuas.png")

# ----------------------------------------------------------------------
# 5. Efecto principal de las categoricas
# ----------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, var in zip(axes3, CAT_COLS):
    preds = []
    for level in cat_levels[var]:
        row = means.to_dict()
        row["ORIGEN"] = df["ORIGEN"].mode()[0]
        row["CAT"] = cat_levels["CAT"][0]
        row[var] = level
        Xg = pd.DataFrame([row])[INPUT_COLS]
        preds.append(modelo.predict(Xg)[0])
    ax.bar([str(l) for l in cat_levels[var]], preds, color="#4c72b0", edgecolor="k")
    ax.set_title(f"Efecto principal de {var}\n(resto en referencia/media)")
    ax.set_ylabel(f"{TARGET} (predicho)")
    ax.grid(alpha=0.3, axis="y")
fig3.tight_layout()
fig3.savefig(f"dependencia_parcial_{TARGET_SLUG}_categoricas.png", dpi=150, bbox_inches="tight")
print(f"Guardado: dependencia_parcial_{TARGET_SLUG}_categoricas.png")
