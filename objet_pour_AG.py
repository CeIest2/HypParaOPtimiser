"""
    Dans ce fichier on va définir les objets qui vont être utilisés pour l'algo génétique

"""
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class MLP(nn.Module):
    """
        Classe qui définit un réseau de neurones MLP 
    """
    def __init__(self, seq_liste_couches):
        super(MLP, self).__init__()
        self.network : nn.Sequential = seq_liste_couches

    
    def forward(self, x):
        return self.network(x)


class Genome:
    """
        Dans cette classe nous allons définir le génome d'u individu

        self.optimizer : str                       => optimiseur du modèle
        self.loss : str                            => fonction de loss du modèle
        self.liste_couche : list[int]              => liste des couches du modèle avec le nombre de neurones par couche
        self.liste_fonction_activation : list[str] => liste des fonctions d'activation pour chaque couche
        self.couche_pytorch : nn.Sequential        => liste des couches du modèle en pytorch
        self.model : MLP                           => modèle pytorch
        self.optimizer_charge : optim              => focntion optimiseur du modèle ( fonction appeler durant l entrainement)
        self.loss_charge : nn                      => fonction de loss du modèle ( fonction appeler durant l entrainement)
        
    """

    def __init__(self,nb_entré : int, nb_sortie : int , param_AG, liste_couche_fixe=-1,
                 liste_fonction_activation_fixe=-1, optimizer_fixe=-1, loss_fixe=-1):
        """
            On peut mettre en paramètre les valeurs les ensembles dans lesquels ont va pouvoir chsoir les valeurs de paramètres pour notre individu
            On peut également passer en paramètre directement els valeurs que dois avoir l'individu, dans le cas ou on ne evut pas faire varier tout les paramètes en même temps
            nb_entré : int     => nombre  d'entré au réseau
            nb_sortie : int    => nombre de sortie du réseau
            param_AG : Para_AG => objet stockant les paramètres de l'algo génétique

                si les variable qui finisse par _fixe ne sont pas présente, elles seront chosie aléatoirement lors de la création de l'individu
            liste_couche_fixe : list[int]              => liste des couches du modèle avec le nombre de neurones par couche
            liste_fonction_activation_fixe : list[str] => liste des fonctions d'activation pour chaque couche
            optimizer_fixe : str                       => optimiseur du modèle
            loss_fixe : str                            => fonction de loss du modèle

        """

        # choix de l'optimiser
        if optimizer_fixe != -1:
            self.optimizer = optimizer_fixe
        else:
            self.optimizer = random.choice(param_AG.liste_optimiser)

        # choix de la foonction de loss
        if loss_fixe != -1:
            self.loss = loss_fixe
        else:
            self.loss = random.choice(param_AG.liste_loss)
        
        # choix des couches et des fonctions d'activation
        if liste_fonction_activation_fixe != -1:
            self.liste_fonction_activation = liste_fonction_activation_fixe

            # on construit aussi les couchesde neurones
            if liste_couche_fixe != -1:
                self.liste_couche = liste_couche_fixe
            else:
                self.liste_couche = [ random.randint(param_AG.range_neurone[0],param_AG.range_neurone[-1]) for i in range(len(self.liste_fonction_activation)) ]
        else:
            if liste_couche_fixe != -1:
                # ce test est la si jamais on a fixé les couches mais pas les fonctions d'activation
                self.liste_fonction_activation = [ random.choice(param_AG.liste_fonction_activation) for i in range(0,len(liste_couche_fixe)) ]
            else:
                self.liste_fonction_activation = [ random.choice(param_AG.liste_fonction_activation) for i in range(0,random.randint(1,param_AG.range_couche[-1])) ]
                 
            # on construit aussi les couches de neurones
            if liste_couche_fixe != -1:
                self.liste_couche = liste_couche_fixe
            else:
                self.liste_couche = [ random.randint(param_AG.range_neurone[0],param_AG.range_neurone[-1]) for i in range(len(self.liste_fonction_activation)) ]

        model_layers =[]
        for i in range(len(self.liste_couche) - 1):

            num_neurons, activation_fn = self.liste_couche[i], self.liste_fonction_activation[i]
            next_num_neurons, _ = self.liste_couche[i + 1], self.liste_couche[i + 1]
            
            model_layers.append(nn.Linear(num_neurons, next_num_neurons))
            
            if activation_fn == 'ReLu':
                model_layers.append(nn.ReLU())
            elif activation_fn == 'LeakyReLu':
                model_layers.append(nn.LeakyReLU())
            elif activation_fn == 'GELU':
                model_layers.append(nn.GELU())
            elif activation_fn == 'Sigmoid':
                model_layers.append(nn.Sigmoid())
            elif activation_fn == 'Tanh':
                model_layers.append(nn.Tanh())

        # On ajoute les couches d'entrées et de sortie du modèle
        model_layers.insert(0, nn.Linear(nb_entré, self.liste_couche[0]))      
        model_layers.append(nn.Linear(self.liste_couche[-1], nb_sortie))

        # création des couches en pytorch
        self.couche_pytorch  = nn.Sequential(*model_layers)
        # création du modèle pytorch
        self.model = MLP(self.couche_pytorch)

        # On charge ici son optimiseur et sa loss fonction
        if self.optimizer == "Adam":
            self.optimizer_charge = optim.Adam(self.model.parameters(), lr=0.01)
        elif self.optimizer == "SGD":
            self.optimizer_charge = optim.SGD(self.model.parameters(), lr=0.01)

        if self.loss == "MSELoss":
            self.loss_charge = nn.MSELoss()
        elif self.loss == "SmoothL1Loss":
            self.loss_charge = nn.SmoothL1Loss()
        elif self.loss == "MAELoss":
            self.loss_charge = nn.L1Loss()
        



class Individu:
    """
        Objet qui définit un individue de la population
    """

    def __init__(self, nb_entré, nb_sortie,param_AG,init=False, liste_couche_fixe=-1,
                 liste_fonction_activation_fixe=-1, optimizer_fixe=-1, loss_fixe=-1):
        self.num_génération : int    = -1   # à modifier lors de la création
        self.list_loss : list[float] = []   # liste des loss pour chaque epoch
        self.first_accu : float      = -1.0 # accu à la finc de la première phase d'entrainement
        self.accu_final : float      = -1.0 # accu à la fin de l'entrainement
        self.type : str              = type # type de l'individu (exemple: "CNN", "RNN", "LSTM", "GRU", "DNN", "MLP")
        self.genome : Genome         = Genome(nb_entré, nb_sortie, param_AG, liste_couche_fixe=liste_couche_fixe,
                                              liste_fonction_activation_fixe=liste_fonction_activation_fixe, 
                                              optimizer_fixe=optimizer_fixe, loss_fixe=loss_fixe)
        self.model : MLP             = self.genome.model
        self.optimizer : str         = self.genome.optimizer    # l optimiser le la loss fonction sont aussi dans le génome
        self.loss : str              = self.genome.loss         # mais c'est plus pratique de les avoir à plusieurs endroits
        self.para_ag : Para_AG       = param_AG           # on met les param de l'AG aussi ici pout simplifier du code plus à d'autres endroits
        self.accu_final_et : float   = -1.0             # accu final sur le data set de test
        self.first_accu_et : float   = -1.0             # accu à la fin de la première phase d'entrainement sur le data set de test

    

class Para_AG:
    """
        On va stocker ici les paramètres de noter simulation pour pouvoir y acceder à tout moment rapidement
    
    """
    def __init__(self, taille_pop=100, nb_entre=5, nb_sortie=1, type="MLP", generation_max = 50,range_couche=[1,6], range_neurone=[1,100], liste_fonction_activation=["ReLu","LeakyReLu","GELU","Sigmoid","Tanh"], liste_optimiser=["Adam","SGD"], liste_loss=["MSELoss","MAELoss","SmoothL1Loss"]):
        """
            taille_pop : int      => taille de la popultation  à créer
            type : str            => type d'individu à créer ( pour l'instant on a que MLP)
            generation_max : int  => nombre de génération maximal que l'on va faire évoluer la population
            range_couche : list   => intervalle de nombre de couches que l'on peut choisir
            range_neurone : list  => intervalle de nombre de neurones que l'on peut choisir
            fonction_activation : list => liste des fonctions d'activation que l'on peut choisir

            permet de stocké les paramètres de AG et pas à avoir à les passer en paramètres partout un par un
        """
        self.nb_entre : int                 = nb_entre         
        self.nb_sortie : int                = nb_sortie
        self.taille_pop : int               = taille_pop
        self.generation_max : int           = generation_max
        self.type : str                     = type
        self.range_couche : list[int]       = range_couche
        self.range_neurone : list[int]      = range_neurone
        self.liste_fonction_activation : list[str]= liste_fonction_activation
        self.liste_optimiser : list[str]    = liste_optimiser
        self.liste_loss : list[str]         = liste_loss
        # pour l'instant tout les param sont mutable et nulle par dans le reste du code on utilsie le dico suivant
        self.dico_para_mutable : dict       = {'nb_couche': True, 'nb_neurones': True, 'fonction_activation': True, 'optimizer': True, 'loss': True}
        # de base on met tout les paramètres mutables mais on peut changer cela à la main après avoir créer l'objet population
        # exemeple : pop = Population(100) 
        #            pop.param.dico_para_mutable['nb_couche'] = False

        self.taux_muta_nb_neurone : float   = 0.2 # taux de mutation du nombre de neurones 
        self.diff_muta_nb_neurone : int     = 7   #les muta se feront dans un interval de plus ou moins 7 neurones

        self.taux_muta_nb_couche : float    =  0.05 # taux de mutation du nombre de couches
        self.diff_muta_nb_couche : int      = 1    # les muta se feront dans un interval de plus ou moins 1 couches

        self.taux_muta_activation_fonction : float = 0.1 # c'est bien le taux pour chaque couche 

        self.taux_muta_optimizer : float = 0.05

        self.taux_muta_loss : float         = 0.05
        self.data_set : Data_set            = Data_set()

class Data_set:
    """ 
        Dans cet objet on va stocker nos différents data set

        Pour l'isntant cette classe est implémenter que pour le test avecla data set de la variable chemin
        mais plus tard on pourra input n'import quel data set et le préparer pour l'entrainement
    """
    def __init__(self):
        chemin = r"..\data_22-11-23.txt"
        df = pd.read_csv(chemin, delimiter=';')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.device = device

        df = df.drop(range(20), axis=0)
        df = df.drop(columns = "Temps")

        new_column_names = ["vit","tension","et_charge","force_roue","temp_bac","courant"]

        df = df.replace(',', '.', regex=True)
        df = df.astype(float)

        scaler_min_max = MinMaxScaler()
        scaled_data = scaler_min_max.fit_transform(df.values)
        data_set = pd.DataFrame(np.copy(scaled_data), columns=new_column_names)
        data_set.columns = new_column_names

        X = data_set.drop(columns=['tension'])
        y = data_set['tension']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Convert your data to tensor 
        self.X_train_tensor : torch.tensor = torch.tensor(X_train.values).float().to(device)
        self.y_train_tensor : torch.tensor = torch.tensor(y_train.values).float().to(device)
        # data set de test
        self.X_test_tensor : torch.tensor = torch.tensor(X_test.values).float().to(device)
        self.y_test_tensor : torch.tensor = torch.tensor(y_test.values).float().to(device)

        # Create DataLoader for training set ( for mini batch training)
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.train_loader = DataLoader(self.train_dataset, batch_size=256, shuffle=True)

        # Create DataLoader for test set
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)




class Population:
    """
        Objet qui définit une population
    """

    def __init__(self, taille_pop=100, nb_entre:int=5, nb_sortie:int=1, type="MLP", generation_max = 50):
        """
            taille_pop : int      => taille de la popultation  à créer
            nb_entre : int        => nombre d'entré au réseau
            nb_sortie : int       => nombre de sortie du réseau
            type : str            => type d'individu à créer ( pour l'instant on a que MLP)
            generation_max : int  => nombre de génération maximal que l'on va faire évoluer la population

            Initialisae un objet population avec lequel on va pouvoir regler les paramètres de l'algo génétique

            return None
        """
        self.taille_pop : int             = taille_pop
        self.generation_max : int         = generation_max
        self.type : str                   = type
        self.liste_indiv : list[Individu] = []
        self.param_AG : Para_AG           = Para_AG(taille_pop=taille_pop, type=type, generation_max=generation_max)
        self.generer_population(nb_entre,nb_sortie)


    def generer_population(self,nb_entre=5,nb_sortie=1):
        """ 
            On génère la population de départ    
            Tout les individu alors générer le sont aléatoirement

            return None
        """
        for i in range(self.taille_pop):
            individu : Individu     = Individu(nb_entre,nb_sortie, self.param_AG, init=True)
            individu.num_génération = 0
            self.liste_indiv.append(individu)

    
