from flask import Flask, request, jsonify
import pickle
import numpy as np
import os  # ✅ Ajouté pour récupérer le PORT

# Charger le modèle et le scaler
with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Standardiser les données
        features = scaler.transform(features)

        # Prédire la classe de la fleur
        prediction = model.predict(features)
        
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # ✅ Récupération du port défini par Render
    app.run(host="0.0.0.0", port=port, debug=True)  # ✅ Adapté pour un déploiement web
