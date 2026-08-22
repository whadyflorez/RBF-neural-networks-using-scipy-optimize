# Bitácora de trabajo futuro
## Modelo RBF multicuádrico con distancia de Gower — ensayo de pirólisis catalítica de biochar

*Documento vivo: ideas identificadas durante el desarrollo del modelo, deliberadamente no implementadas porque no eran necesarias para el problema actual (N=83), pero con valor real como líneas de investigación futura o para cuando se trabaje con datasets más grandes.*

---

## 1. Eficiencia numérica

### 1.1 Cholesky en lugar de `pinv` (la "trampa de las variables *dummy*")

**El problema (verificado empíricamente, no solo en teoría):** al incluir todos los niveles de cada variable categórica como *dummies* junto con un intercepto, se genera una colinealidad exacta (intercepto = suma de los *dummies* de esa categórica) — un autovalor exactamente cero por cada variable categórica. Confirmado con 2 categóricas (ORIGEN, CAT) y con un ejemplo sintético de 4. `np.linalg.cholesky(A)` falla siempre que hay categóricas; el código cae al *fallback* `np.linalg.pinv`, que sí funciona (predicciones idénticas, diferencia ~10⁻¹³) pero es más costoso (SVD vs. factorización + sustituciones triangulares). Contado en una corrida real de H2: **442,084 de 442,084 llamadas** usaron `pinv`, ninguna Cholesky.

**El arreglo:** quitar un nivel de referencia por variable categórica en `build_design_matrix` (la práctica estándar de cualquier regresión con *dummies*). Restaura $A$ positiva definida sin cambiar las predicciones.

**Cuándo retomarlo:** al escalar a datasets más grandes, donde el costo extra de SVD por-llamada empieza a pesar (con N=83 es irrelevante).

### 1.2 Selección incremental de centros (evitar reajustar desde cero en cada paso)

**El problema:** `greedy_select_centers` reconstruye $\Phi$ y resuelve las ecuaciones normales completas para **cada** candidato, en **cada** paso — nunca reutiliza el ajuste del paso anterior.

**El arreglo:** el algoritmo clásico de mínimos cuadrados ortogonales (Gram-Schmidt incremental) evita esto.

**Referencia:** Chen, S., Cowan, C.F.N., & Grant, P.M. (1991). *"Orthogonal least squares learning algorithm for radial basis function networks."* IEEE Transactions on Neural Networks, 2(2), 302–309.

**Cuándo retomarlo:** al escalar N a cientos/miles de puntos (con N=83 el costo actual es de segundos).

### 1.3 LOOCV estocástico para N grande

**La idea:** estimar $\text{tr}(H)$ o la diagonal de $H$ sin calcularla exactamente, vía vectores aleatorios de sondeo — evita el costo de obtener todos los $h_{ii}$ exactos.

**Referencias:**
- Girard, D.A. (1989). *"A fast Monte-Carlo cross-validation procedure for large least squares problems with noisy data."* Numerische Mathematik, 56, 1–23.
- Hutchinson, M.F. (1990). *"A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines."* Communications in Statistics – Simulation and Computation, 19(2), 433–450.
- Bekas, C., Kokiopoulou, E., & Saad, Y. (2007). *"An estimator for the diagonal of a matrix."* Applied Numerical Mathematics. (generaliza de traza a diagonal completa — más fiel al LOOCV punto-por-punto de Rippa).

**Advertencia importante:** el método de Hutchinson/Girard estima $\text{tr}(H)/N$, que es el criterio de **GCV**, no exactamente LOOCV punto-por-punto — son primos cercanos, no idénticos.

**Cuándo retomarlo:** N en el orden de miles.

### 1.4 Reducir el conjunto de candidatos a centro (antes de correr el greedy)

**La idea:** en vez de que el greedy evalúe los N puntos como candidatos en cada paso, preseleccionar un subconjunto vía muestreo ponderado por *ridge leverage scores*.

**Referencia:** El-Alaoui, A., & Mahoney, M.W. (2015). *"Fast randomized kernel ridge regression with statistical guarantees."* NeurIPS 28, 775–783.

**Cuándo retomarlo:** N muy grande, donde evaluar todos los candidatos en cada paso del greedy se vuelve el cuello de botella.

### 1.5 Nota aparte, ya explorada y descartada por ahora (RBF para EDPs, datasets grandes)

La literatura *específica* de RBF para datasets grandes (a diferencia de la de *splines*/GCV) tiende a resolver esto de otra forma: RBF locales, descomposición de dominio, partición de la unidad — evitando el sistema denso N×N por construcción, en vez de aproximar LOOCV estocásticamente. Mencionado por completitud, no es una línea que hayamos empezado a explorar en profundidad.

---

## 2. Búsqueda de hiperparámetros

### 2.1 Optimización Bayesiana en lugar de Latin Hypercube Sampling

**La idea:** en vez de muestrear $(c,\lambda,w_m)$ a ciegas (LHS), construir un modelo sustituto (proceso Gaussiano) sobre los puntos ya evaluados y elegir el siguiente punto maximizando la mejora esperada.

**Referencia:** Snoek, J., Larochelle, H., & Adams, R.P. (2012). *"Practical Bayesian optimization of machine learning algorithms."* NeurIPS 25.

**Cuándo retomarlo:** si el costo por evaluación crece significativamente (p.ej., al escalar N), donde LHS empieza a desperdiciar demasiadas evaluaciones caras.

---

## 3. Selección global de centros (ir más allá del greedy miope)

**El problema de fondo, identificado con el ejemplo pedagógico de 5 puntos:** el greedy solo mira un paso adelante y puede quedar atrapado en un mínimo local de la curva de RMSE-LOOCV — no hay garantía de encontrar el mínimo global.

### 3.1 *Stochastic Greedy*

**La idea:** en cada paso, evaluar solo un subconjunto aleatorio de candidatos (no todos), eligiendo el mejor dentro de esa muestra. Da la misma garantía teórica que el greedy completo (para funciones submodulares), con costo independiente de N.

**Referencias:**
- Mirzasoleiman, B., Badanidiyuru, A., Karbasi, A., Vondrák, J., & Krause, A. (2015). *"Lazier than lazy greedy."* AAAI Conference on Artificial Intelligence, 1812–1818.
- Nemhauser, G.L., Wolsey, L.A., & Fisher, M.L. (1978). *"An analysis of approximations for maximizing submodular set functions – I."* Mathematical Programming, 14(1), 265–294. (el resultado clásico de la garantía $(1-1/e)$)
- Fang, H. et al. *"Efficient mesh deformation using radial basis functions with a grouping-circular-based greedy algorithm."* (aplicación específica a selección de nodos/centros RBF)

**Pregunta abierta, sin resolver:** ¿es el RMSE-LOOCV, como función del conjunto de centros seleccionado, una función submodular? Si lo fuera, el greedy actual ya tendría una garantía teórica de aproximación, no solo evidencia empírica de que funciona bien.

**Precisión importante:** resuelve principalmente el problema de *velocidad*, no necesariamente el de *quedar atrapado en mínimos locales* — sigue sin reconsiderar decisiones pasadas.

### 3.2 *Simulated Annealing* sobre subconjuntos de centros

Acepta movimientos peores con una probabilidad que decrece con el tiempo (temperatura) — permite escapar de mínimos locales genuinos. Uso clásico y bien documentado en selección de centros para redes RBF desde los años 90.

### 3.3 Algoritmos genéticos para selección de centros RBF

Búsqueda poblacional con cruces/mutaciones sobre conjuntos de centros candidatos — otra línea clásica en la literatura de redes RBF.

### 3.4 *Great Deluge Algorithm*

**La idea:** variante determinista de *simulated annealing* — mantiene un "nivel de agua" (umbral) que se endurece con el tiempo; acepta cualquier movimiento que quede por debajo del nivel actual, sin importar si es peor que la solución actual. Un solo parámetro a calibrar (tasa de decaimiento del nivel), más simple que el cronograma de enfriamiento de *simulated annealing*.

**Referencia:** Dueck, G. (1993). *"New optimization heuristics: The great deluge algorithm and the record-to-record travel."* Journal of Computational Physics, 104(1), 86–92.

---

## 4. Pendiente específico de este proyecto (no una técnica general)

### 4.1 Revisar la elección de derivar CO por resta

La decisión de modelar H2, CH4, CO2 directamente y derivar CO = 1−H2−CH4−CO2 se basó en un diagnóstico **lineal** preliminar (regresión lineal con validación cruzada), que mostró a CO como el más difícil de explicar de los 4 gases. Con los 3 modelos RBF (no lineales) ya ajustados, valdría la pena repetir esa comparación de forma no lineal: ajustar también un modelo RBF directo para CO y comparar su RMSE-LOOCV real contra el error del CO derivado — hoy solo se verificó un R² sobre entrenamiento (0.875), no una validación cruzada formal de esta elección.

---

*Última actualización: conversación sobre el ensayo de pirólisis catalítica de biochar, agosto 2026.*
