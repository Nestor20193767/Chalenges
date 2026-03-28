import numpy as np


"""
EN este caso el concepto de Byaes es utilizado para evluar en base a 3 claves:

    1. La probabilidad previa a la prueba de que alguien tenga M (PRIOR)
    2. Si el paciente tiene M que tan probable es que la prueba diga que tiene M (lIKEHOLD)
    3. La posibilidad de un caso especifico de M basado en todos los casos (NORMALIZED)

De esta forma la prueba nos indica que tan real es el resultado sea correcto que tenga M
"""
# Modelo de error de la prueba diagnóstica
# Filas: resultado de la prueba (F, M, H)
# Columnas: diagnóstico real (F, M, H)
diagnostic_model = np.array([[0.8, 0.1, 0.1],
                              [0.1, 0.6, 0.2],
                              [0.1, 0.3, 0.7]])

print("Modelo diagnóstico:")
print(diagnostic_model)

# Índices para mayor claridad en el código
F, M, H = 0, 1, 2

# Distribución de pacientes en la clínica
# 2 ferropénica, 2 megaloblástica, 1 hemolítica
quantities = np.array([2, 2, 1])
print("Distribución de pacientes:", quantities)

p_readM_givenM = diagnostic_model[M][M]
print(f"Verosimilitud p(Z=M | X=M) = {p_readM_givenM}")

p_isM = quantities[M] / sum(quantities)
print(f"Prior p(X=M) = {p_isM}")

# Vector de verosimilitudes: fila M del modelo
p_likelihoods = diagnostic_model[M]
print(f"Verosimilitudes p(Z=M | X=x): {p_likelihoods}")

# Vector de priors
p_isType = quantities / sum(quantities)
print(f"Priors p(X=x): {p_isType}")

# Normalizador
p_readM = np.dot(p_likelihoods, p_isType)
print(f"\nNormalizador p(Z=M) = {p_readM}")

p_isMegaloblastica_givenReadM = p_readM_givenM * p_isM / p_readM
print(f"Posterior p(X=M | Z=M) = {p_isMegaloblastica_givenReadM}")


# ---------------------------------------------------------------------

import matplotlib.pyplot as plt

# Calcular posterior para todos los tipos dado Z=M
posteriors = []
tipos = ['Ferropénica (F)', 'Megaloblástica (M)', 'Hemolítica (H)']

for tipo in [F, M, H]:
    likelihood = diagnostic_model[M][tipo]
    prior = quantities[tipo] / sum(quantities)
    posterior = likelihood * prior / p_readM
    posteriors.append(posterior)
    print(f"p(X={['F','M','H'][tipo]} | Z=M) = {posterior:.4f}")

print(f"\nSuma de posteriors = {sum(posteriors):.2f} (debe ser 1.0)")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colores = ['#C0392B', '#2980B9', '#27AE60']

# Gráfico 1: Priors
priors = quantities / sum(quantities)
axes[0].bar(tipos, priors, color=colores, edgecolor='black', linewidth=0.8)
axes[0].set_title('Prior: Distribución de pacientes\nantes de la prueba', fontsize=13)
axes[0].set_ylabel('Probabilidad', fontsize=11)
axes[0].set_ylim(0, 1)
for i, v in enumerate(priors):
    axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')

# Gráfico 2: Posteriors dado Z=M
axes[1].bar(tipos, posteriors, color=colores, edgecolor='black', linewidth=0.8)
axes[1].set_title('Posterior: Probabilidad real\ndado que la prueba indica M', fontsize=13)
axes[1].set_ylabel('Probabilidad', fontsize=11)
axes[1].set_ylim(0, 1)
for i, v in enumerate(posteriors):
    axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Teorema de Bayes aplicado al diagnóstico de anemia',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('bayes_anemia.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado como bayes_anemia.png")


# Espacio para experimentar
# Modifica las cantidades y observa el impacto en el posterior

# Ejemplo: baja prevalencia de megaloblástica
quantities_new = np.array([18, 1, 1])  # 18 ferropénica, 1 megaloblástica, 1 hemolítica

p_isM_new = quantities_new[M] / sum(quantities_new)
p_isType_new = quantities_new / sum(quantities_new)
p_readM_new = np.dot(diagnostic_model[M], p_isType_new)
p_posterior_new = diagnostic_model[M][M] * p_isM_new / p_readM_new

print(f"Con baja prevalencia de megaloblástica ({p_isM_new:.0%}):")
print(f"p(X=M | Z=M) = {p_posterior_new:.4f} ({p_posterior_new:.1%})")
print()
print("Conclusión: aunque la prueba diga M, con baja prevalencia")
print("la probabilidad real de megaloblástica baja drásticamente.")
print("Esto es el problema de los falsos positivos en enfermedades raras.")
