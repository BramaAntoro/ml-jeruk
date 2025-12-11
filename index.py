import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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


# plt.show()

X = df[["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen","kualitas"]]
y = df["kualitas"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_columns = ["diameter", "berat", "tebal_kulit", "kadar_gula"]
categorical_columns = ["asal_daerah", "musim_panen"]
ordinal_columns = ["warna"]

warna_order = ["hijau", "kuning", "oranye"]
ordinal_order = [warna_order]

preprocessing = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_columns),
        ("ohe", OneHotEncoder(), categorical_columns),
        ("oe", OrdinalEncoder(categories=ordinal_order), ordinal_columns) 
    ]   
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("model", LogisticRegression())
    ]
)
 
# print(model)
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("\n Classification Report : \n", classification_report(y_test, y_pred))
print("\n Confusion Matrix : ", confusion_matrix(y_test, y_pred))

