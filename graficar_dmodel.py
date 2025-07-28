import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración estética
#plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

# Leer datos
df = pd.read_csv("benchmark_d_model.csv")

# 1. Gráfico de Speedup
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='d_model', y='Speedup', hue='Texto', marker='o', linewidth=2.5, markersize=10)
plt.title('Speedup GPU vs CPU por Dimensión del Modelo', pad=20)
plt.xlabel('Dimensión del Modelo (d_model)')
plt.ylabel('Speedup (GPU/CPU)')
plt.xticks(df['d_model'].unique())
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Texto de entrada')
plt.tight_layout()
plt.savefig('speedup.png', dpi=300, bbox_inches='tight')

# 2. Gráfico de Tiempos Comparativos
plt.figure(figsize=(14, 7))
df_melt = df.melt(id_vars=['d_model', 'Texto'], 
                  value_vars=['Tiempo_GPU(ms)', 'Tiempo_CPU(ms)'],
                  var_name='Tipo', 
                  value_name='Tiempo')

sns.barplot(data=df_melt, x='d_model', y='Tiempo', hue='Tipo')
plt.title('Comparación de Tiempos de Ejecución', pad=20)
plt.xlabel('Dimensión del Modelo (d_model)')
plt.ylabel('Tiempo (ms)')
plt.yscale('log')  # Escala logarítmica para mejor visualización
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(title='Dispositivo')
plt.tight_layout()
plt.savefig('tiempos_comparativos.png', dpi=300, bbox_inches='tight')

# 3. Gráfico de Rendimiento (Tokens/segundo)
plt.figure(figsize=(14, 7))
df_melt_perf = df.melt(id_vars=['d_model', 'Texto'], 
                       value_vars=['Tokens/s_GPU', 'Tokens/s_CPU'],
                       var_name='Tipo', 
                       value_name='Rendimiento')

sns.lineplot(data=df_melt_perf, x='d_model', y='Rendimiento', 
             hue='Tipo', style='Texto', markers=True, dashes=False,
             markersize=10, linewidth=2.5)

plt.title('Rendimiento en Tokens/segundo', pad=20)
plt.xlabel('Dimensión del Modelo (d_model)')
plt.ylabel('Tokens/segundo')
plt.xticks(df['d_model'].unique())
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Configuración')
plt.tight_layout()
plt.savefig('rendimiento.png', dpi=300, bbox_inches='tight')

# 4. Gráfico Combinado (Subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Subplot 1: Speedup
sns.lineplot(data=df, x='d_model', y='Speedup', hue='Texto', 
             ax=ax1, marker='o', linewidth=2.5)
ax1.set_title('Speedup por Dimensión del Modelo')
ax1.set_xlabel('d_model')
ax1.set_ylabel('Speedup (x)')
ax1.grid(True)

# Subplot 2: Rendimiento GPU
sns.lineplot(data=df, x='d_model', y='Tokens/s_GPU', hue='Texto',
             ax=ax2, marker='s', linewidth=2.5)
ax2.set_title('Rendimiento GPU por Dimensión')
ax2.set_xlabel('d_model')
ax2.set_ylabel('Tokens/segundo (GPU)')
ax2.grid(True)

plt.tight_layout()
plt.savefig('combinado.png', dpi=300, bbox_inches='tight')

print("✅ Gráficos generados:")
print("- speedup.png")
print("- tiempos_comparativos.png")
print("- rendimiento.png")
print("- combinado.png")