from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


model_path = 'mejor_modelo_rrff(1).pkl'
scaler_path = 'scaler(1).pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
   
    df = pd.DataFrame([data])
   
    df_scaled = scaler.transform(df)
 
    prediction = model.predict(df_scaled)
   
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)