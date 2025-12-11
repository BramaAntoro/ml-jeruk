import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('jeruk_balance_500.csv')

bagus = df[df['kualitas'] == 'Bagus']
sedang = df[df['kualitas'] == 'Sedang']
jelek = df[df['kualitas'] == 'Jelek']

# diamater vs berat
plt.figure(figsize=(6,5))

plt.scatter(bagus['diameter'], bagus['berat'], s=100, alpha=0.7, color='blue', label='Bagus')
plt.scatter(sedang['diameter'], sedang['berat'], s=100, alpha=0.7, color='orange', label='Sedang')
plt.scatter(jelek['diameter'], jelek['berat'], s=100, alpha=0.7, color='red', label='Jelek')

plt.xlabel('Diamater')
plt.ylabel("Berat")
plt.title("Diamater vs Berat")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)


# tebal kulit vs kadar gula
plt.figure(figsize=(6,5))

plt.scatter(bagus['tebal_kulit'], bagus['kadar_gula'], s=100, alpha=0.7, color='blue', label='Bagus')
plt.scatter(sedang['tebal_kulit'], sedang['kadar_gula'], s=100, alpha=0.7, color='orange', label='Sedang')
plt.scatter(jelek['tebal_kulit'], jelek['kadar_gula'], s=100, alpha=0.7, color='red', label='Jelek')

plt.xlabel('Tebal Kulit')
plt.ylabel("Kadar Gula")
plt.title("Tebal Kulit vs Kadar Gula")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)


plt.show()

