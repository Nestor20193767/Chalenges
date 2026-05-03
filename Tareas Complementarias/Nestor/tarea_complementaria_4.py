"""
============================================================
TAREA COMPLEMENTARIA 4 - Reconocimiento de Patrones (UPCH)
Árboles de Decisión, Bagging, Random Forest y Ensambles
============================================================

Curso   : Reconocimiento de Patrones
Docente : Msc. Emilio Ochoa Alva
Tema    : Árboles de decisión y ensambles

Este archivo resuelve COMPLETAMENTE los dos problemas de la tarea.
Cada sección incluye:
  - Explicación conceptual detallada (en los comentarios)
  - Implementación numérica verificable
  - Resultados impresos con interpretación
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SEPARADOR VISUAL
# ─────────────────────────────────────────────────────────────────────────────
SEP  = "=" * 70
SEP2 = "-" * 70

# ============================================================
#  PROBLEMA 1
#  Bagging, Random Forest y varianza del ensamble
# ============================================================

print(SEP)
print("  PROBLEMA 1: BAGGING, RANDOM FOREST Y VARIANZA DEL ENSAMBLE")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 1a) Idea central del bagging y diferencias con boosting
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Apartado 1a) Bagging vs Boosting ---\n")

respuesta_1a = """
IDEA CENTRAL DEL BAGGING (Bootstrap AGGregating)
─────────────────────────────────────────────────
El bagging parte de la siguiente observación:
  Si pudiéramos entrenar m modelos INDEPENDIENTES y promediar sus
  predicciones, la varianza del promedio se reduciría en un factor m.
  El problema es que solo tenemos UN conjunto de datos.

Solución:  Simular "nuevos" datasets mediante BOOTSTRAP SAMPLING.
  1. Dado el dataset original D con n ejemplos,
     generar m conjuntos D_1, ..., D_m muestreando n ejemplos
     CON REEMPLAZO desde D.
  2. Entrenar un modelo independiente h_i en cada D_i.
  3. Para clasificación: predicción final = voto mayoritario
     Para regresión    : predicción final = promedio

El bootstrap preserva el tamaño del dataset y produce muestras
"casi independientes" (en promedio, cada muestra bootstrap contiene
~63.2 % de los datos originales únicos).

DIFERENCIAS BAGGING vs BOOSTING
────────────────────────────────
  CRITERIO          │  BAGGING                   │  BOOSTING
  ──────────────────┼────────────────────────────┼───────────────────────────
  Entrenamiento     │  PARALELO: cada modelo      │  SECUENCIAL: cada modelo
                    │  se entrena de forma        │  depende del anterior.
                    │  independiente.             │
  ──────────────────┼────────────────────────────┼───────────────────────────
  Objetivo          │  Reducir la VARIANZA        │  Reducir el SESGO (bias).
                    │  (combate sobreajuste).     │  (combate subajuste).
  ──────────────────┼────────────────────────────┼───────────────────────────
  Pesos muestras    │  Uniformes en cada          │  Adaptativos: se aumenta
                    │  bootstrap.                 │  el peso de los ejemplos
                    │                             │  mal clasificados.
  ──────────────────┼────────────────────────────┼───────────────────────────
  Riesgo principal  │  Si los modelos base tienen │  Si se ejecuta demasiadas
                    │  mucho SESGO (subajuste),   │  rondas puede SOBREAJUSTAR,
                    │  el ensamble también lo     │  pues pondera cada vez más
                    │  tendrá (bagging no lo      │  ejemplos ruidosos/atípicos.
                    │  corrige).                  │
  ──────────────────┼────────────────────────────┼───────────────────────────
  Ejemplo clásico   │  Random Forest              │  AdaBoost / XGBoost
"""
print(respuesta_1a)

# ─────────────────────────────────────────────────────────────────────────────
# 1b) Análisis de la fórmula de varianza del ensamble
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Apartado 1b) Análisis de Var(ȳ) ---\n")

# La fórmula clave estudiada en clase (diapositiva 53):
#
#   Var(ȳ) = ( (1 - ρ) / m  +  ρ ) · σ²
#
# Derivación intuitiva:
#   Si y_1, ..., y_m tienen varianza σ² y correlación pairwise ρ, entonces:
#   Var(∑ y_i) = m·σ² + m(m-1)·ρ·σ²
#   Var(ȳ)     = Var(∑ y_i) / m²
#              = σ²/m  +  (m-1)/m · ρ · σ²
#              ≈ (1-ρ)/m · σ²  +  ρ · σ²
#              = ( (1-ρ)/m + ρ ) · σ²

print("Fórmula de la varianza del ensamble:")
print("  Var(ȳ) = [ (1 - ρ) / m  +  ρ ] · σ²\n")

# ── Pregunta 1: ¿bagging reduce sesgo, varianza o ambos? ────────────────────
print("  [P1] ¿El bagging reduce principalmente el sesgo, la varianza o ambos?")
print("""
  RESPUESTA: El bagging reduce principalmente la VARIANZA.

  Demostración con la fórmula:
    E[ȳ] = E[y_i]  (el promedio de valores con la misma esperanza conserva
                    la esperanza, no la cambia)

  ➜  El SESGO del ensamble = sesgo de cualquier predictor individual.
     No mejora ni empeora.

  ➜  La VARIANZA del ensamble ≤ varianza individual σ²,
     porque el término (1-ρ)/m → 0 cuando m → ∞
     y el término ρ·σ² actúa como piso mínimo.

  En el contexto del hospital:
    Si los árboles individuales sobreajustan los ECG de entrenamiento
    (alta varianza), el bagging los suaviza promediando sus predicciones.
""")

# ── Pregunta 2: ¿qué ocurre cuando m aumenta? ────────────────────────────────
print("  [P2] ¿Qué ocurre con Var(ȳ) cuando m aumenta?")
print()

# Visualización numérica
sigma2 = 1.0   # varianza individual normalizada
rho    = 0.3   # correlación ejemplo

print(f"  Ejemplo numérico: σ² = {sigma2}, ρ = {rho}\n")
print(f"  {'m':>6}  {'Var(ȳ)':>10}  {'Reducción vs m=1':>20}")
print(f"  {'-'*40}")
var_m1 = None
for m in [1, 2, 5, 10, 25, 50, 100, 500, 10_000]:
    var_ensemble = ((1 - rho) / m + rho) * sigma2
    if m == 1:
        var_m1 = var_ensemble
    reduccion = (1 - var_ensemble / var_m1) * 100
    print(f"  {m:>6}  {var_ensemble:>10.4f}  {reduccion:>18.1f}%")

print("""
  INTERPRETACIÓN:
    - Al aumentar m, el término (1-ρ)/m tiende a 0.
    - La varianza converge al piso ρ·σ² (no puede bajar de ahí).
    - Después de m ≈ 100 árboles, las ganancias adicionales son mínimas.
    - Por eso en Random Forest se usan típicamente 100–500 árboles.
""")

# ── Pregunta 3: ¿qué ocurre si ρ es muy alto? ───────────────────────────────
print("  [P3] ¿Qué ocurre si ρ es muy alto?")
print()

print(f"  {'ρ':>6}  {'Var(ȳ) con m=100':>20}  {'Var(ȳ) con m=∞':>18}")
print(f"  {'-'*50}")
m_grande = 100
for rho_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    var_100 = ((1 - rho_val) / m_grande + rho_val) * sigma2
    var_inf = rho_val * sigma2          # límite cuando m → ∞
    print(f"  {rho_val:>6.1f}  {var_100:>20.4f}  {var_inf:>18.4f}")

print("""
  INTERPRETACIÓN:
    - Si ρ → 1: todos los árboles son casi idénticos (aprenden lo mismo).
      La varianza del ensamble → σ² (ninguna mejora). Agregar más árboles
      no ayuda: Var(ȳ) ≈ σ² independientemente de m.
    - Si ρ → 0: árboles completamente independientes.
      La varianza → 0 cuando m → ∞.  (Caso ideal, irrealizable en práctica.)

  En el caso médico:
    Si todos los árboles reciben casi el mismo subconjunto de features de ECG,
    aprenderán patrones similares → alta correlación → poca reducción de varianza.
    Por eso Random Forest introduce aleatoriedad extra.
""")

# ── Pregunta 4: Random Forest y su efecto sobre ρ ────────────────────────────
print("  [P4] Random Forest: efecto sobre ρ y la varianza del ensamble")
print("""
  En un árbol de bagging estándar, todos los árboles compiten por los
  MISMOS atributos dominantes (ej. la feature más informativa del ECG).
  Esto hace que muchos árboles elijan el mismo split en la raíz
  → árboles muy similares → alta correlación ρ.

  Random Forest añade aleatoriedad:
    En cada nodo, en lugar de evaluar TODAS las d features, se elige
    un subconjunto aleatorio de tamaño d' (típicamente d' = √d).
    Solo ese subconjunto se usa para buscar el mejor split.

  Efecto matemático:
    - La feature dominante NO siempre está disponible → los árboles
      eligen splits diferentes → los árboles son más DISTINTOS entre sí.
    - ρ disminuye respecto al bagging puro.
    - Según la fórmula: Var(ȳ) = ((1-ρ)/m + ρ)·σ², al disminuir ρ
      disminuye tanto el término (1-ρ)/m como el piso ρ·σ².
    - Resultado: menor varianza del ensamble con el mismo m.

  Costo: cada árbol individual es ligeramente peor (no siempre usa
  la mejor feature), pero el ensamble global es mejor gracias a la
  menor correlación entre árboles.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 1c) Error Out-Of-Bag (OOB)
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Apartado 1c) Error Out-Of-Bag (OOB) ---\n")

print("""
DEFINICIÓN DEL ERROR OOB
──────────────────────────
Cuando se genera una muestra bootstrap D_i de tamaño n (con reemplazo),
aproximadamente el 36.8 % de los ejemplos originales NO son seleccionados.
Estos ejemplos se llaman Out-Of-Bag (OOB) para el árbol i.

Cálculo del error OOB:
  1. Para cada ejemplo x_j del dataset original,
     identificar todos los árboles h_i que NO usaron x_j en su entrenamiento
     (es decir, x_j ∈ OOB_i).
  2. Hacer la predicción de x_j usando SOLO esos árboles.
  3. Comparar con la etiqueta real y_j.
  4. El error OOB global = tasa de error promedio sobre todos los x_j.

¿POR QUÉ ES VÁLIDO COMO ESTIMACIÓN DE GENERALIZACIÓN?
───────────────────────────────────────────────────────
Cada ejemplo x_j es evaluado únicamente por árboles que NO lo vieron
durante el entrenamiento → equivale a un test set "invisible" para cada árbol.

Ventajas frente a validación cruzada (cross-validation):
  - No requiere separar explícitamente datos de validación
    (maximiza uso de datos de entrenamiento, útil en medicina con n pequeño).
  - Se calcula "gratis" durante el proceso de entrenamiento.
  - Suele ser una estimación casi tan buena como la validación cruzada 5-10 fold.

Limitación:
  - No es completamente equivalente a un hold-out, porque los árboles OOB
    son solo ~63 % del ensamble total (menor capacidad que el modelo final).
    El error OOB tiende a ser ligeramente pesimista.
""")

# Simulación conceptual del error OOB
print("  Simulación conceptual del error OOB:")
np.random.seed(42)
n = 15                        # tamaño del dataset (igual al Problema 2)
m_trees = 10                  # número de árboles

oob_counts = np.zeros(n, dtype=int)  # cuántas veces cada ejemplo es OOB
for t in range(m_trees):
    bootstrap_idx = np.random.choice(n, size=n, replace=True)
    in_bag  = set(bootstrap_idx)
    oob_idx = set(range(n)) - in_bag
    for idx in oob_idx:
        oob_counts[idx] += 1

print(f"\n  Dataset: n={n} ejemplos, m={m_trees} árboles bootstrap")
print(f"  {'Ejemplo':>8}  {'Veces OOB':>10}  {'% árboles sin verlo':>22}")
for i in range(n):
    pct = oob_counts[i] / m_trees * 100
    print(f"  {i:>8}  {oob_counts[i]:>10}  {pct:>21.1f}%")
print(f"\n  Promedio de veces OOB por ejemplo: {oob_counts.mean():.1f} / {m_trees}")
print(f"  Fracción promedio OOB: {oob_counts.mean()/m_trees:.3f}  (teórico: 0.368)")

# ─────────────────────────────────────────────────────────────────────────────
# 1d) Probabilidad de no ser seleccionado en bootstrap
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Apartado 1d) Probabilidad de no ser seleccionado ---\n")

print("""
DERIVACIÓN DE  (1 - 1/n)^n ≈ e^{-1} ≈ 0.368
─────────────────────────────────────────────
Sea un dataset con n ejemplos. En una muestra bootstrap, seleccionamos
n ejemplos CON REEMPLAZO (es decir, n sorteos independientes).

Probabilidad de que un ejemplo ESPECÍFICO sea elegido en un sorteo:
    P(elegido en 1 sorteo) = 1/n

Probabilidad de que NO sea elegido en ese sorteo:
    P(no elegido en 1 sorteo) = 1 - 1/n

Probabilidad de que NO sea elegido en NINGUNO de los n sorteos:
    P(OOB) = (1 - 1/n)^n

Límite cuando n → ∞ (definición del número e):
    lim_{n→∞} (1 - 1/n)^n = e^{-1} ≈ 0.3679
""")

# Verificación numérica
print("  Verificación numérica para distintos n:")
print(f"  {'n':>8}  {'(1-1/n)^n':>12}  {'Error vs e^-1':>16}")
e_inv = np.exp(-1)
print(f"  e^(-1) = {e_inv:.6f}")
print()
for n_val in [5, 10, 15, 20, 50, 100, 1000, 10000]:
    prob_oob = (1 - 1/n_val) ** n_val
    error_rel = abs(prob_oob - e_inv) / e_inv * 100
    print(f"  {n_val:>8}  {prob_oob:>12.6f}  {error_rel:>14.2f}%")

print("""
INTERPRETACIÓN Y RELACIÓN CON EL ERROR OOB
───────────────────────────────────────────
Este resultado significa que, para cualquier dataset de tamaño n,
cada muestra bootstrap deja fuera aproximadamente el 36.8 % de los
ejemplos originales.

En términos prácticos (para el hospital con n=15 pacientes):
  - Cada árbol "no ve" en promedio: 15 × 0.368 ≈ 5-6 pacientes.
  - Esos 5-6 pacientes son el conjunto OOB de ese árbol.
  - Para cada paciente, en promedio ~36.8 % de los árboles lo tendrán
    como OOB → se puede calcular el error OOB de forma confiable.

Consecuencia para el error OOB:
  - Hay suficientes árboles que "no vieron" cada ejemplo para hacer
    una predicción ensemble representativa.
  - Cuanto mayor sea m (número de árboles), más estable es el error OOB.
""")


# ============================================================
#  PROBLEMA 2
#  Árbol de decisión, índice Gini y profundidad máxima
# ============================================================

print()
print(SEP)
print("  PROBLEMA 2: ÁRBOL DE DECISIÓN, ÍNDICE GINI, PROF. MÁX. = 2")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

data = {
    "Paciente": list("ABCDEFGHIJKLMNO"),
    "x1": [5.5, 7.4, 5.9, 9.9, 6.9, 6.8, 4.1, 1.3, 4.5, 0.5, 5.9, 9.3, 1.0, 0.4, 2.7],
    "x2": [0.5, 1.1, 0.2, 0.1,-0.1,-0.3, 0.3,-0.2, 0.4, 0.0,-0.1,-0.2, 0.1, 0.1,-0.5],
    "x3": [4.5, 3.6, 3.4, 0.8, 0.6, 5.1, 5.1, 1.8, 2.0, 2.3, 4.4, 3.2, 2.8, 4.3, 4.2],
    "y":  [  2,   0,   2,   0,   2,   2,   1,   1,   0,   1,   0,   0,   1,   1,   1],
}
df = pd.DataFrame(data)

print("\nDataset original (15 pacientes, 3 features, 3 clases):")
print(df.to_string(index=False))

# Distribución de clases en el nodo raíz
y = df["y"].values
clases, conteos = np.unique(y, return_counts=True)
n_total = len(y)

print(f"\nDistribución de clases:")
for c, cnt in zip(clases, conteos):
    nombre = {0: "sin evento adverso", 1: "riesgo intermedio", 2: "evento adverso"}
    print(f"  Clase {c} ({nombre[c]}): {cnt} pacientes  ({cnt/n_total*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def gini_impurity(labels):
    """
    Calcula el índice de Gini de un conjunto de etiquetas.

    Fórmula: Gini = 1 - Σ p_k²
    donde p_k = fracción de muestras de la clase k.

    El Gini mide IMPUREZA:
      - Gini = 0   →  nodo perfectamente puro (una sola clase)
      - Gini = 0.5 →  máxima impureza para 2 clases (50/50)
      - Gini < 0.667 para 3 clases
    """
    if len(labels) == 0:
        return 0.0
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


def gini_split(labels_left, labels_right):
    """
    Calcula la impureza de Gini ponderada DESPUÉS de un split.

    Gini_split = (n_L/n)·Gini_L + (n_R/n)·Gini_R
    """
    n = len(labels_left) + len(labels_right)
    if n == 0:
        return 0.0
    w_l = len(labels_left)  / n
    w_r = len(labels_right) / n
    return w_l * gini_impurity(labels_left) + w_r * gini_impurity(labels_right)


def delta_gini(labels_parent, labels_left, labels_right):
    """
    Reducción de impureza (Information Gain Gini):
    ΔGini = Gini_padre - Gini_split_ponderado

    Queremos MAXIMIZAR este valor al elegir el split.
    """
    return gini_impurity(labels_parent) - gini_split(labels_left, labels_right)


def find_best_split(X_col, y_labels, feature_name, min_samples_leaf=2):
    """
    Encuentra el MEJOR umbral (threshold) para una feature continua.

    Estrategia:
      1. Ordenar los valores únicos de la feature.
      2. Evaluar como threshold el PUNTO MEDIO entre valores consecutivos.
      3. Calcular ΔGini para cada threshold.
      4. Devolver el threshold con mayor ΔGini.

    Parámetro min_samples_leaf: mínimo de muestras en cada hoja
    (la tarea pide no evaluar splits con < 2 muestras en un nodo).
    """
    sorted_vals = np.sort(np.unique(X_col))
    best_threshold  = None
    best_delta_gini = -np.inf
    best_gini_left  = None
    best_gini_right = None
    best_nl = best_nr = 0
    results = []

    for i in range(len(sorted_vals) - 1):
        threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
        mask_left  = X_col <= threshold
        mask_right = X_col >  threshold
        y_left  = y_labels[mask_left]
        y_right = y_labels[mask_right]

        # Respetar restricción de tamaño mínimo
        if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
            continue

        dg = delta_gini(y_labels, y_left, y_right)
        gl = gini_impurity(y_left)
        gr = gini_impurity(y_right)
        results.append({
            "feature":    feature_name,
            "threshold":  threshold,
            "n_left":     len(y_left),
            "n_right":    len(y_right),
            "gini_left":  gl,
            "gini_right": gr,
            "delta_gini": dg,
        })
        if dg > best_delta_gini:
            best_delta_gini = dg
            best_threshold  = threshold
            best_gini_left  = gl
            best_gini_right = gr
            best_nl = len(y_left)
            best_nr = len(y_right)

    return best_threshold, best_delta_gini, best_gini_left, best_gini_right, \
           best_nl, best_nr, results


# ─────────────────────────────────────────────────────────────────────────────
# 2a) Construcción del árbol – NODO RAÍZ (profundidad 0 → 1)
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP2)
print("  2a) CONSTRUCCIÓN DEL ÁRBOL (profundidad máxima = 2)")
print(SEP2)

X = df[["x1", "x2", "x3"]].values
y = df["y"].values
features = ["x1", "x2", "x3"]

# ── Calcular Gini raíz ──────────────────────────────────────────────────────
gini_root = gini_impurity(y)
print(f"\n[NODO RAÍZ]  n={n_total} muestras")
print(f"  Distribución: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  Gini(raíz) = 1 - (5/15)² - (6/15)² - (4/15)²")
print(f"             = 1 - {(5/15)**2:.4f} - {(6/15)**2:.4f} - {(4/15)**2:.4f}")
print(f"             = {gini_root:.4f}")

print("\nEVALUACIÓN DE TODOS LOS SPLITS EN LA RAÍZ:")
print("=" * 90)

all_splits_root = []
for feat_idx, feat_name in enumerate(features):
    X_col = X[:, feat_idx]
    thresh, dg, gl, gr, nl, nr, all_results = find_best_split(X_col, y, feat_name)

    print(f"\n  Feature {feat_name}:")
    print(f"  {'Threshold':>12}  {'n_L':>5}  {'Gini_L':>8}  {'n_R':>5}  {'Gini_R':>8}  {'ΔGini':>8}")
    print(f"  {'-'*60}")
    for r in all_results:
        marker = " ◄ MEJOR" if abs(r["threshold"] - thresh) < 1e-9 else ""
        print(f"  {r['threshold']:>12.3f}  {r['n_left']:>5}  {r['gini_left']:>8.4f}  "
              f"{r['n_right']:>5}  {r['gini_right']:>8.4f}  {r['delta_gini']:>8.4f}{marker}")

    all_splits_root.append((feat_name, thresh, dg, gl, gr, nl, nr))

# Seleccionar el mejor split global para la raíz
best_root = max(all_splits_root, key=lambda t: t[2])
feat_root, thresh_root, dg_root, gl_root, gr_root, nl_root, nr_root = best_root

print(f"\n{'='*90}")
print(f"  MEJOR SPLIT EN LA RAÍZ:")
print(f"  Feature   = {feat_root}")
print(f"  Threshold = {thresh_root:.3f}")
print(f"  ΔGini     = {dg_root:.4f}")
print(f"  Nodo L (≤{thresh_root:.3f}): n={nl_root}, Gini={gl_root:.4f}")
print(f"  Nodo R (> {thresh_root:.3f}): n={nr_root}, Gini={gr_root:.4f}")

# Dividir el dataset según el mejor split de la raíz
feat_root_idx = features.index(feat_root)
mask_L1 = X[:, feat_root_idx] <= thresh_root
mask_R1 = X[:, feat_root_idx] >  thresh_root

X_L1, y_L1 = X[mask_L1], y[mask_L1]
X_R1, y_R1 = X[mask_R1], y[mask_R1]

pacientes_L1 = df["Paciente"].values[mask_L1]
pacientes_R1 = df["Paciente"].values[mask_R1]

print(f"\n  Pacientes rama IZQUIERDA ({feat_root} ≤ {thresh_root:.3f}): {list(pacientes_L1)}")
print(f"  Clases: {list(y_L1)}")
print(f"\n  Pacientes rama DERECHA ({feat_root} > {thresh_root:.3f}): {list(pacientes_R1)}")
print(f"  Clases: {list(y_R1)}")

# ── Nodo izquierdo (profundidad 1 → 2) ─────────────────────────────────────
print(f"\n{SEP2}")
print(f"  NODO IZQUIERDO (profundidad 1):  {feat_root} ≤ {thresh_root:.3f}")
print(f"  n={nl_root},  Clases: {dict(Counter(y_L1))},  Gini={gl_root:.4f}")

print(f"\n  Evaluando splits en nodo IZQUIERDO:")
print(f"  {'Feature':>6}  {'Best Thresh':>12}  {'ΔGini':>8}")
print(f"  {'-'*35}")

all_splits_L1 = []
for feat_idx, feat_name in enumerate(features):
    X_col = X_L1[:, feat_idx]
    thresh, dg, gl, gr, nl, nr, _ = find_best_split(X_col, y_L1, feat_name)
    if thresh is not None:
        all_splits_L1.append((feat_name, thresh, dg, gl, gr, nl, nr))
        print(f"  {feat_name:>6}  {thresh:>12.3f}  {dg:>8.4f}")

if all_splits_L1:
    best_L1 = max(all_splits_L1, key=lambda t: t[2])
    f_L1, t_L1, dg_L1, gl_L1, gr_L1, nl_L1, nr_L1 = best_L1
    print(f"\n  → Mejor split en nodo izquierdo: {f_L1} ≤ {t_L1:.3f}  (ΔGini={dg_L1:.4f})")
else:
    print("  → Nodo puro o no hay splits válidos. Se convierte en hoja.")
    f_L1, t_L1, dg_L1 = None, None, 0.0

# ── Nodo derecho (profundidad 1 → 2) ────────────────────────────────────────
print(f"\n{SEP2}")
print(f"  NODO DERECHO (profundidad 1):  {feat_root} > {thresh_root:.3f}")
print(f"  n={nr_root},  Clases: {dict(Counter(y_R1))},  Gini={gr_root:.4f}")

print(f"\n  Evaluando splits en nodo DERECHO:")
print(f"  {'Feature':>6}  {'Best Thresh':>12}  {'ΔGini':>8}")
print(f"  {'-'*35}")

all_splits_R1 = []
for feat_idx, feat_name in enumerate(features):
    X_col = X_R1[:, feat_idx]
    thresh, dg, gl, gr, nl, nr, _ = find_best_split(X_col, y_R1, feat_name)
    if thresh is not None:
        all_splits_R1.append((feat_name, thresh, dg, gl, gr, nl, nr))
        print(f"  {feat_name:>6}  {thresh:>12.3f}  {dg:>8.4f}")

if all_splits_R1:
    best_R1 = max(all_splits_R1, key=lambda t: t[2])
    f_R1, t_R1, dg_R1, gl_R1, gr_R1, nl_R1, nr_R1 = best_R1
    print(f"\n  → Mejor split en nodo derecho: {f_R1} ≤ {t_R1:.3f}  (ΔGini={dg_R1:.4f})")
else:
    print("  → Nodo puro o no hay splits válidos. Se convierte en hoja.")
    f_R1, t_R1, dg_R1 = None, None, 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Resumen del árbol completo
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  ÁRBOL DE DECISIÓN COMPLETO (profundidad 2)")
print(SEP)

# Obtener las hojas finales
feat_L1_idx = features.index(f_L1) if f_L1 else None
feat_R1_idx = features.index(f_R1) if f_R1 else None

# Hojas del nodo izquierdo
if f_L1 is not None:
    mask_LL = X_L1[:, feat_L1_idx] <= t_L1
    mask_LR = X_L1[:, feat_L1_idx] >  t_L1
    y_LL = y_L1[mask_LL]
    y_LR = y_L1[mask_LR]
    pac_LL = pacientes_L1[mask_LL]
    pac_LR = pacientes_L1[mask_LR]
else:
    y_LL, y_LR = y_L1, np.array([])
    pac_LL, pac_LR = pacientes_L1, np.array([])

# Hojas del nodo derecho
if f_R1 is not None:
    mask_RL = X_R1[:, feat_R1_idx] <= t_R1
    mask_RR = X_R1[:, feat_R1_idx] >  t_R1
    y_RL = y_R1[mask_RL]
    y_RR = y_R1[mask_RR]
    pac_RL = pacientes_R1[mask_RL]
    pac_RR = pacientes_R1[mask_RR]
else:
    y_RL, y_RR = y_R1, np.array([])
    pac_RL, pac_RR = pacientes_R1, np.array([])


print(f"""
  Estructura del árbol:
  ─────────────────────────────────────────────────────────────────
  RAÍZ: {feat_root} ≤ {thresh_root:.3f}  ?
        ├── SI (≤ {thresh_root:.3f})  → NODO IZQUIERDO: {f_L1} ≤ {t_L1:.3f}  ?
        │       ├── SI → Hoja LL
        │       └── NO → Hoja LR
        └── NO (> {thresh_root:.3f})  → NODO DERECHO: {f_R1} ≤ {t_R1:.3f}  ?
                ├── SI → Hoja RL
                └── NO → Hoja RR
  ─────────────────────────────────────────────────────────────────
""")

# ─────────────────────────────────────────────────────────────────────────────
# 2b) Descripción de cada hoja final
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("  2b) DESCRIPCIÓN DE LAS HOJAS FINALES")
print(SEP2)

def describe_leaf(nombre, pacientes, y_leaf):
    """Imprime la descripción completa de una hoja."""
    if len(y_leaf) == 0:
        print(f"\n  [{nombre}] VACÍA")
        return
    n_leaf     = len(y_leaf)
    dist       = Counter(y_leaf)
    clase_pred = max(dist, key=dist.get)
    gini_leaf  = gini_impurity(y_leaf)
    prob_dict  = {c: cnt/n_leaf for c, cnt in dist.items()}
    nombre_clase = {0: "sin evento adverso", 1: "riesgo intermedio", 2: "evento adverso"}

    print(f"\n  [Hoja {nombre}]")
    print(f"    Pacientes      : {list(pacientes)}")
    print(f"    n              : {n_leaf}")
    print(f"    Distribución   : {dict(dist)}")
    print(f"    Probabilidades : { {c: f'{p:.3f}' for c,p in prob_dict.items()} }")
    print(f"    Clase predicha : {clase_pred}  ({nombre_clase[clase_pred]})")
    print(f"    Gini           : {gini_leaf:.4f}")

describe_leaf("LL (izq-izq)", pac_LL, y_LL)
describe_leaf("LR (izq-der)", pac_LR, y_LR)
describe_leaf("RL (der-izq)", pac_RL, y_RL)
describe_leaf("RR (der-der)", pac_RR, y_RR)

# ─────────────────────────────────────────────────────────────────────────────
# 2c) Clasificación de nuevos pacientes
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP2)
print("  2c) CLASIFICACIÓN DE NUEVOS PACIENTES")
print(SEP2)

# Nuevos pacientes
nuevos = {
    "x_a": np.array([4.1, -0.1, 2.2]),
    "x_b": np.array([6.1,  0.4, 1.3]),
}

print("""
  Cómo se recorre el árbol para clasificar un nuevo ejemplo:
  1. Ir a la raíz: evaluar si la feature del split raíz ≤ threshold.
  2. Seguir la rama correspondiente (izquierda o derecha).
  3. En el nodo interno del nivel 1: evaluar la feature de ese nodo.
  4. Llegar a una hoja: la predicción es la clase más frecuente en esa hoja.
  5. La probabilidad empírica p(c|x,T) = fracción de esa clase en la hoja.
""")

def classify_sample(x, feat_root, thresh_root, f_L1, t_L1, f_R1, t_R1,
                    y_LL, y_LR, y_RL, y_RR,
                    pac_LL, pac_LR, pac_RL, pac_RR, features):
    """
    Recorre el árbol de profundidad 2 y devuelve la hoja, clase predicha
    y distribución de probabilidades empíricas.
    """
    feat_root_idx = features.index(feat_root)
    nombre_hoja = None

    if x[feat_root_idx] <= thresh_root:
        # Rama izquierda
        feat_L1_idx = features.index(f_L1)
        if x[feat_L1_idx] <= t_L1:
            y_leaf = y_LL;  pac = pac_LL;  nombre_hoja = "LL"
        else:
            y_leaf = y_LR;  pac = pac_LR;  nombre_hoja = "LR"
    else:
        # Rama derecha
        feat_R1_idx = features.index(f_R1)
        if x[feat_R1_idx] <= t_R1:
            y_leaf = y_RL;  pac = pac_RL;  nombre_hoja = "RL"
        else:
            y_leaf = y_RR;  pac = pac_RR;  nombre_hoja = "RR"

    n_leaf = len(y_leaf)
    dist   = Counter(y_leaf)
    clase_pred = max(dist, key=dist.get)
    probs  = {c: cnt/n_leaf for c, cnt in dist.items()}
    # Asegurar que todas las clases aparezcan (con prob 0 si no existen)
    for c in [0, 1, 2]:
        if c not in probs:
            probs[c] = 0.0
    return nombre_hoja, clase_pred, probs, pac

nombre_clase = {0: "sin evento adverso", 1: "riesgo intermedio", 2: "evento adverso"}

for nombre_px, x_nuevo in nuevos.items():
    print(f"\n  Paciente {nombre_px}: x1={x_nuevo[0]}, x2={x_nuevo[1]}, x3={x_nuevo[2]}")
    print(f"  {'─'*50}")

    # Recorrido paso a paso
    feat_root_idx = features.index(feat_root)
    cond_raiz = x_nuevo[feat_root_idx] <= thresh_root
    print(f"  Paso 1 – Raíz: {feat_root}={x_nuevo[feat_root_idx]} ≤ {thresh_root:.3f}?  → {'SÍ' if cond_raiz else 'NO'}")

    if cond_raiz:
        feat_L1_idx = features.index(f_L1)
        cond_L1 = x_nuevo[feat_L1_idx] <= t_L1
        print(f"  Paso 2 – Nodo izq: {f_L1}={x_nuevo[feat_L1_idx]} ≤ {t_L1:.3f}?  → {'SÍ' if cond_L1 else 'NO'}")
    else:
        feat_R1_idx = features.index(f_R1)
        cond_R1 = x_nuevo[feat_R1_idx] <= t_R1
        print(f"  Paso 2 – Nodo der: {f_R1}={x_nuevo[feat_R1_idx]} ≤ {t_R1:.3f}?  → {'SÍ' if cond_R1 else 'NO'}")

    hoja, clase, probs, pac = classify_sample(
        x_nuevo, feat_root, thresh_root,
        f_L1, t_L1, f_R1, t_R1,
        y_LL, y_LR, y_RL, y_RR,
        pac_LL, pac_LR, pac_RL, pac_RR,
        features
    )
    print(f"\n  RESULTADO:")
    print(f"    Hoja alcanzada  : {hoja}")
    print(f"    Pacientes en hoja: {list(pac)}")
    print(f"    Clase predicha  : {clase}  ({nombre_clase[clase]})")
    print(f"    p(c=0 | x, T)   : {probs[0]:.3f}")
    print(f"    p(c=1 | x, T)   : {probs[1]:.3f}")
    print(f"    p(c=2 | x, T)   : {probs[2]:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2d) ¿Por qué limitar la profundidad evita sobreajuste?
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP2)
print("  2d) PROFUNDIDAD MÁXIMA Y SOBREAJUSTE EN PROBLEMAS BIOMÉDICOS")
print(SEP2)

print("""
¿POR QUÉ LIMITAR LA PROFUNDIDAD EVITA EL SOBREAJUSTE?
───────────────────────────────────────────────────────
Un árbol sin restricción de profundidad puede crecer hasta que cada hoja
contenga UN SOLO ejemplo de entrenamiento (Gini = 0 en todos los nodos).
Ese árbol memoriza el dataset pero generaliza muy mal.

PROBLEMA ESPECÍFICO EN BIOMEDICINA:
  1. Datos escasos: los estudios clínicos suelen tener n pequeño
     (en este caso, solo 15 pacientes).
  2. Ruido clínico: las mediciones de ECG tienen variabilidad intrínseca
     (artefactos, variación inter-paciente, errores de medición).
  3. Alta dimensionalidad: múltiples biomarcadores, muchos son ruidosos.

CON PROFUNDIDAD = 2:
  - Cada hoja tiene varios pacientes (no solo uno).
  - La predicción es el PROMEDIO estadístico de esa región.
  - No memoriza fluctuaciones ruidosas individuales.
  - Las reglas de decisión son simples e interpretables por médicos.

RELACIÓN CON SESGO-VARIANZA:
  ╔═══════════╦══════════════════╦══════════════════╗
  ║ Parámetro ║  Árbol profundo  ║  Árbol limitado  ║
  ╠═══════════╬══════════════════╬══════════════════╣
  ║ Sesgo     ║  Bajo            ║  Mayor           ║
  ║ Varianza  ║  Alta            ║  Menor           ║
  ║ Resultado ║  Sobreajuste     ║  Mejor generaliz.║
  ╚═══════════╩══════════════════╩══════════════════╝

ALTERNATIVAS COMPLEMENTARIAS (mencionadas en clase):
  - Pruning (poda post-entrenamiento)
  - Bagging / Random Forest (reduce varianza sin perder tanto sesgo)
  - min_samples_leaf: no crear hojas con < k muestras
  - max_features: reducir features por nodo (usado en Random Forest)

EJEMPLO CONCRETO:
  Con depth=2, nuestra hoja LL usa varios pacientes → el modelo predice
  la clase modal con cierta incertidumbre (Gini > 0 en algunas hojas).
  Un árbol completo (depth=4 para n=15) tendría hojas de 1 paciente →
  cualquier nuevo paciente similar pero no idéntico podría ser clasificado
  erroneamente.
""")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("  RESUMEN EJECUTIVO DE RESULTADOS")
print(SEP)

print(f"""
  PROBLEMA 1 – BAGGING Y VARIANZA
  ─────────────────────────────────────────────────────────────────────
  a) Bagging = bootstrap + promediar. Paralelo, reduce varianza.
     Boosting = secuencial, pondera errores, reduce sesgo.

  b) Fórmula Var(ȳ) = ((1-ρ)/m + ρ)·σ²:
     - Bagging SOLO reduce varianza (no sesgo).
     - Al aumentar m: Var(ȳ) → ρ·σ²  (piso mínimo alcanzable).
     - ρ alto: los árboles son similares → poco beneficio del ensamble.
     - Random Forest: selección aleatoria de features → menor ρ → menor Var.

  c) OOB: cada árbol no ve ~36.8% de los datos → evalúa esos ejemplos
     como si fueran test set → estimación sin separar datos de validación.

  d) (1-1/n)^n → e^(-1) ≈ 0.368 para n grande:
     ~36.8% de ejemplos quedan OOB en cada árbol bootstrap.

  PROBLEMA 2 – ÁRBOL DE DECISIÓN GINI (profundidad 2)
  ─────────────────────────────────────────────────────────────────────
  Gini raíz = {gini_root:.4f}

  ÁRBOL FINAL:
  ┌─ Raíz: {feat_root} ≤ {thresh_root:.2f}?  (ΔGini = {dg_root:.4f})
  │   ├─[SÍ] Nodo: {f_L1} ≤ {t_L1:.2f}?  (ΔGini = {dg_L1:.4f})
  │   │   ├─[SÍ] Hoja LL:  n={len(y_LL)},  dist={dict(Counter(y_LL))},  pred={max(Counter(y_LL), key=Counter(y_LL).get) if len(y_LL)>0 else 'N/A'},  Gini={gini_impurity(y_LL):.4f}
  │   │   └─[NO] Hoja LR:  n={len(y_LR)},  dist={dict(Counter(y_LR))},  pred={max(Counter(y_LR), key=Counter(y_LR).get) if len(y_LR)>0 else 'N/A'},  Gini={gini_impurity(y_LR):.4f}
  │   └─[NO] Nodo: {f_R1} ≤ {t_R1:.2f}?  (ΔGini = {dg_R1:.4f})
  │       ├─[SÍ] Hoja RL:  n={len(y_RL)},  dist={dict(Counter(y_RL))},  pred={max(Counter(y_RL), key=Counter(y_RL).get) if len(y_RL)>0 else 'N/A'},  Gini={gini_impurity(y_RL):.4f}
  │       └─[NO] Hoja RR:  n={len(y_RR)},  dist={dict(Counter(y_RR))},  pred={max(Counter(y_RR), key=Counter(y_RR).get) if len(y_RR)>0 else 'N/A'},  Gini={gini_impurity(y_RR):.4f}

  CLASIFICACIÓN NUEVOS PACIENTES:
""")

for nombre_px, x_nuevo in nuevos.items():
    hoja, clase, probs, _ = classify_sample(
        x_nuevo, feat_root, thresh_root,
        f_L1, t_L1, f_R1, t_R1,
        y_LL, y_LR, y_RL, y_RR,
        pac_LL, pac_LR, pac_RL, pac_RR,
        features
    )
    print(f"  {nombre_px}: Hoja={hoja}, "
          f"Clase={clase} ({nombre_clase[clase]}), "
          f"p(0)={probs[0]:.2f}, p(1)={probs[1]:.2f}, p(2)={probs[2]:.2f}")

print(f"\n{SEP}")
print("  FIN DE LA TAREA COMPLEMENTARIA 4")
print(SEP)
