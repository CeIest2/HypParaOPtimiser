"""
    Dans ce fichier nous allons réaliser les fonctions qui permettent de faire les croisement entre individus
"""

import random
import objet_pour_AG as AG


def croisement(indiv1 : AG.Individu, indiv2 : AG.Individu, param_AG : AG.Para_AG) -> AG.Individu:
    """
        Fonction qui va réaliser le croisement entre deux individus
        Les deux individus passés en paramètre ne sont pas modifiés
    """


    # On crée le génome de l'individu

    # On crée le génome de l'individu
    new_liste_couche : list[int] = []
    new_liste_fonction_activation : list[str] = []
    new_loss : str               = ""
    new_optimiser : str          = ""

    nb_couche = random.choice([len(indiv1.genome.liste_couche), len(indiv2.genome.liste_couche)])

    # Pour le croisement on fait la moyenne des valeurs des deux indiv et si on dépasse le nombre
    # de couche d'un indiv on recopie les valeurs de l'autre indiv

    for i in range(nb_couche):
        if i > len(indiv1.genome.liste_couche) - 1:
            new_liste_couche.append(indiv2.genome.liste_couche[i])
            new_liste_fonction_activation.append(indiv2.genome.liste_fonction_activation[i])
        elif i > len(indiv2.genome.liste_couche) - 1:
            new_liste_couche.append(indiv1.genome.liste_couche[i])
            new_liste_fonction_activation.append(indiv1.genome.liste_fonction_activation[i])
        else:
            new_liste_couche.append((indiv1.genome.liste_couche[i] + indiv2.genome.liste_couche[i]) // 2)
            new_liste_fonction_activation.append(random.choice([indiv1.genome.liste_fonction_activation[i], indiv2.genome.liste_fonction_activation[i]]))

    # On choisit aléatoirement la fonction de loss et l'optimiseur
    new_loss : str      = random.choice([indiv1.genome.loss, indiv2.genome.loss])
    new_optimiser : str = random.choice([indiv1.genome.optimizer, indiv2.genome.optimizer])

    # On crée l'individu
    new_indiv = AG.Individu(param_AG.nb_entre, param_AG.nb_sortie, param_AG, liste_couche_fixe=new_liste_couche, liste_fonction_activation_fixe=new_liste_fonction_activation, loss_fixe=new_loss, optimizer_fixe=new_optimiser)
    return new_indiv