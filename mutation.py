
"""
    Dans ce fichier ou va coder les fonctions qui vont permettre de muter les individus

"""

import random
import objet_pour_AG as AG



def mutation(indiv : AG.Individu, gene_a_muter = ['nb_couche', 'nb_neurones', 'fonction_activation', 'optimiser', 'fonction_loss'] ) -> AG.Individu:
    """
        fonction qui va lancer la mutation d'un individu
        ! un individu mutant est créé mais l'individu passé en paramètre de cette fonctino n'est pas modifié !
    
    """

    ###### Mutation du nombre de couche ######

    param_AG = indiv.para_ag
    if random.random() > param_AG.taux_muta_nb_couche :
        mutation = 0
    else:
        if random.random() > 0.5:
            if len(indiv.genome.liste_couche) < param_AG.range_couche[-1]:
                mutation = 1

            elif len(indiv.genome.liste_couche) > param_AG.range_couche[0]:
                mutation = -1
            else:
                mutation = 0
        else:
            mutation = 0
    
    ####### Préparation de la mutation du nombre de neurones et des fonctions d'activation #######

    # cas ou on a retiré une couche
    couche_mutant = []
    if mutation < 0:
        # si le nombre de couche a changer on sélectionne aléatoirement la couche à supprimer
        ind_retirer = random.randint(0, len(indiv.genome.liste_couche) - 1)
        i = 0
        # On crée la liste avec les nombres de neurones dans chaque couches
        for nb_neurone in indiv.genome.liste_couche:
            if nb_neurone != ind_retirer:
                couche_mutant.append(nb_neurone)
            else:
                i += 1
        # On créer mainteanant la liste des fonctions d'activation
        i = 0
        liste_fonction_activation = []
        for fonction_activation in indiv.genome.liste_fonction_activation:
            if fonction_activation != ind_retirer:
                liste_fonction_activation.append(fonction_activation)
            i += 1
    
    # cas ou on a ajouté une couche
    elif mutation > 0:
        # On décide d'ou insérer une couche dans le réseau
        ind_insérer = random.randint(0, len(indiv.genome.liste_couche) - 1)

        # On fait d'abord des copies des listes et on va insert la ouvelle couches juste après
        liste_fonction_activation = [ i for i in indiv.genome.liste_fonction_activation]
        liste_fonction_activation.insert(ind_insérer, random.choice(param_AG.liste_fonction_activation))

        # On fait la même chose pour les nombres de neurones
        couche_mutant = [ i for i in indiv.genome.liste_couche]
        couche_mutant.insert(ind_insérer, random.randint(param_AG.range_couche[0], param_AG.range_couche[-1]))

    # cas ou on a pas touché au nombre de couche

    else:
        couche_mutant = [ i for i in indiv.genome.liste_couche]
        liste_fonction_activation = [ i for i in indiv.genome.liste_fonction_activation]    

    ####### Mutation du nombre de neurones par couches #######
    for couche in range(len(couche_mutant)):
        if random.random() < param_AG.taux_muta_nb_neurone:
            muta  = random.randint(-param_AG.diff_muta_nb_neurone, param_AG.diff_muta_nb_neurone)
            
            if couche_mutant[couche] + muta < param_AG.range_neurone[0]:
                couche_mutant[couche] = param_AG.range_neurone[0]
            elif couche_mutant[couche] + muta > param_AG.range_neurone[-1]:
                couche_mutant[couche] = param_AG.range_neurone[-1]
            else:
                couche_mutant[couche] += muta


    ####### Mutation des fonctions d'activation #######
    for i in range(len(liste_fonction_activation)):
        if random.random() < param_AG.taux_muta_activation_fonction:
            liste_fonction_activation[i] = random.choice(param_AG.liste_fonction_activation)

    ####### Mutation de l'optimiseur #######
    if random.random() < param_AG.taux_muta_optimizer:
        muta_optimiser = random.choice(param_AG.liste_optimiser)
    else: 
        muta_optimiser = indiv.genome.optimizer

    ####### Mutation de la fonction de loss #######
    if random.random() < param_AG.taux_muta_loss:
        muta_fonction_loss = random.choice(param_AG.liste_loss)
    else:
        muta_fonction_loss = indiv.genome.loss

    # On crée l'individu mutant
    indiv_mutant = AG.Individu(param_AG.nb_entre, param_AG.nb_sortie, param_AG, liste_couche_fixe=couche_mutant, liste_fonction_activation_fixe=liste_fonction_activation, optimizer_fixe=muta_optimiser, loss_fixe=muta_fonction_loss)

    return indiv_mutant