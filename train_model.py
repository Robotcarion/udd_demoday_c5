import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el archivo CSV
file_path = r"C:\Users\KPaz\Downloads\BOOTCAMP\GitHub\proyecto_a_presentar\dataset_limpio_2.csv"  # Reemplaza con la ruta correcta
df = pd.read_csv(file_path)

# Agregar la columna 'Edad_Obtencion_Oscar'
df['Edad_Obtencion_Oscar'] = df['year_of_award'] - df['year_birth']

# Convertir variables categóricas a numéricas
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Dividir el dataframe en X e y
X = df.drop(columns=['race_ethnicity'])
y = df['race_ethnicity']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
model_boost = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
).fit(X_train_scaled, y_train)

# Guardar el modelo y el scaler
model_path = 'modelo_boost.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model_boost, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)