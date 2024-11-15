"""
    Dans ce fichier on va faire les fonction qui permettent de faire les entraîenemnt des modèles
"""

import objet_pour_AG as ag
import pandas as pd
import time
import torch
import torch.nn as nn
import numpy as np

# Fonction de loss perso pour évaluer les modèles avec l'écart type des prédiction par rapport aux valeurs réelles
class StdDevLoss(nn.Module):
    def __init__(self):
        super(StdDevLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        std_dev = torch.std(diff)
        return std_dev



def train_population(population : ag.Population, train_part=0) -> None:
    """
        Fonction qui va entraîner tous les modèles de la population
        train_part = 0 pour l'entrainement initial et 1 pour l'entrainement final
    """
    X_train_loader = population.param_AG.data_set.train_loader
    X_test_loader  = population.param_AG.data_set.test_loader
    device         = population.param_AG.data_set.device   # "cpu" ou "cuda"
 

    for indiv in population.liste_indiv:
        if indiv.first_accu != -1.0 and train_part == 0:
            continue
        if indiv.accu_final != -1.0 and train_part == 1:
            continue
        model = indiv.model
        # Define the loss function
        criterion = indiv.genome.loss_charge.to(device)
        optimizer = indiv.genome.optimizer_charge
        # List to store loss values
        loss_values = []

        if train_part == 0:
            nb_epoch = 2
        else:
            nb_epoch = 10

        for epoch in range(nb_epoch):
            model.train()
            epoch_loss = 0.0
            for batch_X_train, batch_y_train in X_train_loader:
                outputs = model(batch_X_train)
                loss = criterion(outputs, batch_y_train.unsqueeze(1)).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Average loss for the epoch
            loss_values.append(epoch_loss / len(X_train_loader))

        indiv.loss = [i for i in loss_values]

        model.eval()

        # Evaluate the model on the test data
        test_loss = 0.0
        with torch.no_grad():
            for batch_X_test, batch_y_test in X_test_loader:
                predictions = model(batch_X_test)
                batch_loss = criterion(predictions, batch_y_test.unsqueeze(1)).item()
                test_loss += batch_loss
        test_loss /= len(X_test_loader)

        if train_part == 0:
            indiv.first_accu = test_loss
        else:
            indiv.accu_final = test_loss

        # Evaluate with standard deviation loss
        criterion_stddev = StdDevLoss()
        test_loss_et = 0.0
        with torch.no_grad():
            for batch_X_test, batch_y_test in X_test_loader:
                predictions = model(batch_X_test)
                batch_loss_et = criterion_stddev(predictions, batch_y_test.unsqueeze(1)).item()
                test_loss_et += batch_loss_et
        test_loss_et /= len(X_test_loader)

        if train_part == 0:
            indiv.first_accu_et = test_loss_et
            print(f"train fini et loss de fin {indiv.first_accu}")
            print(f"train fini et loss de fin avec ecart type {test_loss_et}")
        else:
            indiv.accu_final_et = test_loss_et
            print(f"train fini et loss de fin {indiv.accu_final}")
            print(f"train fini et loss de fin avec ecart type {test_loss_et}")
        print(f"train part {train_part}")
        print(f"fonction de loss {indiv.genome.loss} et optimizer {indiv.genome.optimizer}")
        print(f"#######################")