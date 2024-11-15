import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import onnxruntime as rt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('20240613_f105b9.csv', delimiter=';')

# Supprimer les 50 premières lignes
df = df.iloc[50:]
df = df[["vitesse", "tension", "courant", "frein", "omega", "couple", "accel", "theta"]]

# Prétraitement des données sans normalisation
df = df.replace(',', '.', regex=True)
df = df.astype(float)
new_column_names = df.columns

y_label = pd.DataFrame(np.copy(df.values[:, 2]), columns=['courant'])
data_set = pd.DataFrame(np.copy(df.values), columns=new_column_names)

X = data_set.drop(columns=['courant'])
y = data_set['courant']

X_all_tensor = torch.tensor(X.values).float().to(device)
y_all_tensor = torch.tensor(y.values).float().to(device)


# Charger le modèle ONNX
sess = rt.InferenceSession('model_courant.onnx')

# Définir les entrées pour le modèle
input_name = sess.get_inputs()[0].name

# Faire des prédictions avec le modèle ONNX
def predict_onnx(X):
    X_onnx = X.cpu().numpy()
    pred_onnx = np.empty((X.shape[0], 1))
    for i in range(X.shape[0]):
        pred_onnx[i] = sess.run(None, {input_name: X_onnx[i].reshape(1, -1)})[0]
    return torch.from_numpy(pred_onnx).to(device)

# Prédire les valeurs cibles pour l'ensemble du dataset de données
batch_size = 32
predictions_all = []
for i in range(0, len(X_all_tensor), batch_size):
    X_batch = X_all_tensor[i:i+batch_size]
    pred_batch = predict_onnx(X_batch)
    predictions_all.append(pred_batch)
predictions_all = torch.cat(predictions_all, dim=0)

# Afficher les données réelles et les prédictions du modèle sur un graphique
plt.plot(y.values, label='Données réelles')
plt.plot(predictions_all.cpu().detach().numpy(), label='Prédictions du modèle ONNX')
plt.xlabel('Temps')
plt.ylabel('Valeur cible')
plt.legend()
plt.show()
