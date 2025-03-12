import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Étape 1 : Charger et préparer les données
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Gestion des valeurs manquantes (dans le dataset Iris, il n'y en a pas, mais on applique un traitement par sécurité)
X.fillna(X.median(), inplace=True)

# Transformation des variables (standardisation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en jeu d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entraîner le modèle KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Évaluer les performances
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print("\nMatrice de confusion:\n", conf_matrix)
print("\nRapport de classification:\n", class_report)

# Sauvegarder le modèle
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
