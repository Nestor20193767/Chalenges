# =========================== Problema 1 ===================================

# --- Configuración de Datos Iniciales ---

# 1. Probabilidad a priori: P(S) 
# Probabilidad de tener sepsis (1 de cada 100)
p_sepsis = 1 / 100 

# 2. Probabilidad de estar sano: P(H)
# Es el complemento de tener sepsis
p_sano = 1 - p_sepsis

# 3. Sensibilidad del sistema: P(+|S)
# Probabilidad de dar positivo si el paciente REALMENTE tiene sepsis
sensibilidad = 0.95

# 4. Especificidad del sistema: P(-|H)
# Probabilidad de dar negativo si el paciente está REALMENTE sano
especificidad = 0.95

# 5. Tasa de Falsos Positivos: P(+|H)
# Probabilidad de dar positivo estando sano (1 - especificidad)
p_falso_positivo = 1 - especificidad


# --- Cálculo del Teorema de Bayes ---

# Paso A: Calcular el numerador de la fórmula (Verdaderos Positivos)
# Probabilidad de tener sepsis Y que el test salga positivo
verdaderos_positivos = sensibilidad * p_sepsis

# Paso B: Calcular los Falsos Positivos
# Probabilidad de estar sano Y que el test salga positivo por error
falsos_positivos = p_falso_positivo * p_sano

# Paso C: Probabilidad Total de dar positivo: P(+)
# Es la suma de todos los casos en los que el sistema dice "POSITIVO"
p_total_positivo = verdaderos_positivos + falsos_positivos

# Paso D: Aplicación final del Teorema de Bayes: P(S|+)
# Dividimos los casos de éxito real entre el total de alertas positivas
probabilidad_real_sepsis = verdaderos_positivos / p_total_positivo


# --- Salida de Resultados ---

print(f"--- Análisis de Resultados ---")
print(f"Probabilidad de Sepsis en la población: {p_sepsis * 100}%")
print(f"Total de positivos detectados por el sistema: {p_total_positivo:.4f}")
print("-" * 30)
print(f"PROBABILIDAD REAL DE TENER SEPSIS: {probabilidad_real_sepsis:.4%}")


# =========================== Problema 2 ===================================

# Definimos una funcion que nos permita realizar esos calculos tomando encuenta todo lo desfcrito antes
"""
--- ORIGEN MATEMÁTICO DE LAS FÓRMULAS ---
    
    1. Función de Densidad de Probabilidad (PDF):
       p(x) = 1 / (b - a)  para  a <= x <= b
       p(x) = 0            en cualquier otro caso

    2. Nacimiento de la MEDIA (Esperanza Matemática):
       La media se define como la integral de x por la PDF:
       E[X] = ∫[a,b] x * (1 / (b - a)) dx
       
       Pasos:
       - Sacamos la constante: (1 / (b-a)) * ∫[a,b] x dx
       - Integramos x: (1 / (b-a)) * [x² / 2] evaluado de a a b
       - Evaluamos: (1 / (b-a)) * (b² - a²) / 2
       - Factorizamos (b² - a²) como (b - a)(b + a):
         ((b - a)(b + a)) / (2 * (b - a))
       - Simplificamos: (a + b) / 2

    3. Nacimiento de la VARIANZA:
       Se define como Var(X) = E[X²] - (E[X])²
       
       Primero hallamos E[X²]:
       E[X²] = ∫[a,b] x² * (1 / (b - a)) dx = (1 / (b-a)) * [x³ / 3] de a a b
       E[X²] = (b³ - a³) / (3 * (b - a))
       Simplificando (diferencia de cubos): (a² + ab + b²) / 3

       Luego, restamos el cuadrado de la media:
       Var(X) = [(a² + ab + b²) / 3] - [(a + b)² / 4]
       
       Al resolver la resta de fracciones y simplificar el binomio al cuadrado,
       llegamos a la forma compacta:
       Var(X) = (b - a)² / 12
"""

def analizar_señal_ecg(a, b):
    """
    Calcula la media y varianza de una señal con distribución uniforme.
    a: Límite inferior del voltaje (mV)
    b: Límite superior del voltaje (mV)
    """
    
    # 1. Cálculo de la Media (Punto medio del intervalo)
    media = (a + b) / 2
    
    # 2. Cálculo de la Varianza (Medida de dispersión)
    # La fórmula es (ancho del intervalo al cuadrado) / 12
    varianza = ((b - a)**2) / 12
    
    # 3. Desviación estándar (opcional, para entender la dispersión en mV)
    desviacion = varianza**0.5
    
    return media, varianza, desviacion

# Ejemplo de uso:
# Supongamos una señal de ECG que oscila entre -0.5 mV y 0.5 mV
v_min = -0.5
v_max = 0.5

mu, var, std = analizar_señal_ecg(v_min, v_max)

print(f"--- Análisis de Señal Uniforme [{v_min}, {v_max}] mV ---")
print(f"Media (E[X]): {mu:.4f} mV")
print(f"Varianza (Var(X)): {var:.4f} mV²")
print(f"Desviación Estándar: {std:.4f} mV")
