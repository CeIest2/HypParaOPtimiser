"""

    Dans ce fichier nous allons réaliser les fonction qui permettent de faire la sélection des modèles
"""



import random
import objet_pour_AG as AG


def selection(population : AG.Population, nb_indiv_selectionne : int, etat_training=0) -> AG.Population:
    """
        Fonction qui va réaliser la sélection des individus de la population
    """
    # On trie la population en fonction de la loss
    if etat_training == 0:
        population.liste_indiv = sorted(population.liste_indiv, key=lambda x: x.first_accu_et, reverse=False)
    # On sélectionne les meilleurs individus
    else:
        population.liste_indiv = sorted(population.liste_indiv, key=lambda x: (x.accu_final_et >= 0, x.accu_final_et), reverse=False)
    population.liste_indiv = population.liste_indiv[:nb_indiv_selectionne]
    return population