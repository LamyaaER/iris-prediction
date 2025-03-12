import streamlit as st
import requests

st.title("Prédiction de la Classe des Fleurs (Iris)")

# Entrées utilisateur
p1 = st.number_input("Longueur du sépale", min_value=0.0, max_value=10.0, step=0.1, value=5.1)
p2 = st.number_input("Largeur du sépale", min_value=0.0, max_value=10.0, step=0.1, value=3.5)
p3 = st.number_input("Longueur du pétale", min_value=0.0, max_value=10.0, step=0.1, value=1.4)
p4 = st.number_input("Largeur du pétale", min_value=0.0, max_value=10.0, step=0.1, value=0.2)

if st.button("Prédire"):
    url = "http://127.0.0.1:5000/predict" 
    data = {"features": [p1, p2, p3, p4]}
    response = requests.post(url, json=data)  

    if response.status_code == 200:
        result = response.json()
        st.write(f"Classe prédite : {result['prediction']}")
    else:
        st.write("Erreur dans la prédiction.")
