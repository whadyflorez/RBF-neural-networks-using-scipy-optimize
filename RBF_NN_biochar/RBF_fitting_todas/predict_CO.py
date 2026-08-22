"""
predict_CO.py
==============
CO no se modela directamente (ver justificacion en el informe tecnico:
era, por lejos, el gas mas dificil de explicar linealmente desde las
entradas). Se deriva de la restriccion composicional exacta:

    CO = 1 - H2 - CH4 - CO2

Este script reajusta los 3 modelos de gas (mismos hiperparametros ya
validados en fit_H2.py / fit_CH4.py / fit_CO2.py) y expone una funcion
predict_gas_completo(X_new) que devuelve los 4 gases -incluido CO
derivado- garantizando que sumen exactamente 1 para cualquier punto nuevo.
"""
import numpy as np
import pandas as pd
from rbf_gower import RBFApproximator

RUTA_EXCEL = "dataset_biochar_reorganizado.xlsx"
CAT_COLS = ["ORIGEN", "CAT"]
CONT_COLS = ["CARGA %", "TR", "N2", "T"]
INPUT_COLS = CAT_COLS + CONT_COLS

df = pd.read_excel(RUTA_EXCEL, sheet_name="Datos", header=4)
df = df.rename(columns={"T (°C)": "T"})
X = df[INPUT_COLS].copy()

# Hiperparametros validados en cada fit_<gas>.py -- NO se reutiliza la
# misma configuracion entre gases.
CONFIGS = {
    "H2":  dict(c_range=(0.01, 20.0),  lam_range=(1e-8, 10.0), cat_weight_range=(0.001, 300.0),
                n_search_points=500, search_seed=123),
    "CH4": dict(c_range=(0.01, 20.0),  lam_range=(1e-10, 10.0), cat_weight_range=(0.0001, 300.0),
                n_search_points=350, search_seed=7),
    "CO2": dict(c_range=(0.01, 20.0),  lam_range=(1e-10, 10.0), cat_weight_range=(0.0001, 300.0),
                n_search_points=250, search_seed=3),
}

modelos = {}
for gas, cfg in CONFIGS.items():
    print(f"Ajustando {gas}...")
    m = RBFApproximator(cat_cols=CAT_COLS, max_centers=40, verbose=False, **cfg)
    m.fit(X, df[gas].values)
    modelos[gas] = m
    print(f"  RMSE-LOOCV({gas}) = {m.rmse_loocv_:.5f}")


def predict_gas_completo(X_new):
    """
    X_new: DataFrame con columnas INPUT_COLS.
    Devuelve DataFrame con columnas H2, CH4, CO2, CO (derivado), que suman
    exactamente 1 fila por fila.
    """
    h2 = modelos["H2"].predict(X_new)
    ch4 = modelos["CH4"].predict(X_new)
    co2 = modelos["CO2"].predict(X_new)
    co = 1.0 - h2 - ch4 - co2
    return pd.DataFrame({"H2": h2, "CH4": ch4, "CO2": co2, "CO": co}, index=X_new.index)


if __name__ == "__main__":
    # Verificacion sobre los datos de entrenamiento: comparar CO derivado
    # vs. CO real observado (no se uso CO real para ajustar nada).
    pred = predict_gas_completo(X)
    suma = pred.sum(axis=1)
    print("\nVerificacion de cierre composicional (debe ser 1.0 en todas las filas):")
    print("  min suma =", suma.min(), "  max suma =", suma.max())

    co_real = df["CO"].values
    co_derivado = pred["CO"].values
    rmse_co = np.sqrt(np.mean((co_real - co_derivado) ** 2))
    r2_co = 1 - np.sum((co_real - co_derivado) ** 2) / np.sum((co_real - co_real.mean()) ** 2)
    print(f"\nCO derivado vs. CO real observado (sobre datos de entrenamiento):")
    print(f"  RMSE = {rmse_co:.5f}   R^2 = {r2_co:.4f}")
    print("  (para comparar limpio contra un CO modelado directamente, habria que "
          "hacerlo tambien via LOOCV -- este numero es solo un chequeo rapido de "
          "consistencia, no una validacion cruzada formal de la eleccion de derivar CO).")

    # ejemplo de uso sobre un punto nuevo
    ejemplo = pd.DataFrame([{
        "ORIGEN": "amazonas", "CAT": "CaO", "CARGA %": 0.05,
        "TR": 0.5, "N2": 2.2, "T": 600,
    }])
    print("\nEjemplo, punto nuevo:")
    print(predict_gas_completo(ejemplo))
