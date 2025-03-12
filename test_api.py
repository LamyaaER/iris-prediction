import requests

# URL de ton API déployée sur Render
url = "https://iris-prediction-65h7.onrender.com/predict"

# En-têtes HTTP
headers = {"Content-Type": "application/json"}

# Données de test (features d'une fleur Iris)
data = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Envoi de la requête POST
response = requests.post(url, json=data, headers=headers)

# Affichage de la réponse JSON
print(response.json())
