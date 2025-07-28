import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")

# Gráfico de tiempos
plt.figure(figsize=(10, 5))
plt.bar(df['Texto'], df['Tiempo_GPU(ms)'], label='GPU')
plt.bar(df['Texto'], df['Tiempo_CPU(ms)'], bottom=df['Tiempo_GPU(ms)'], label='CPU')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Tiempo (ms)')
plt.title('Comparación GPU vs CPU')
plt.legend()
plt.tight_layout()
plt.savefig('tiempos.png')
plt.show()