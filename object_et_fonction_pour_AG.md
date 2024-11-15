# On va définir ici toutes les fonctions néccessaire à l'algo génétique

!!! Pendant les teste bine vérifier à chque fois si les objets sont passer en référence ou en valeurs



# objets : 

## Classe Individu : 
    comprend les paramètres d'un individu:
        - numéro génération de création
        - note de performance ( résultat fonction de fitnesse) default : -1
        -dico de tout les paramètres que l'on peut prendre en compte dans l'ag
        -liste des para qui sont prit en compte actuelement dans l'ag ( True ou False pour savoir ceux modifier durant l'ag)
        -liste des valeurs de la fonctions de perte durant les n premier batch ( n sera un para de la simulation de l'ag)
        - valeur de performance du modèle ( réel pour les modèles qui ont finit l'entraînement ou calculer à partir du réseau de prédiction ou bout de m batch)
        - le coeff qui calculer à partir de la diff de perf par rapport au autres modèle ( coeff pour la mutation)
        - Booléen pour savoir si l'indiv à déjà fait un train complet

## Classe génome

    bblablaablabal
    ablabalbla

génome pour un MLP:
-nombre de couches
-nombre de neurones dans cahue couche
-fonction d'activation de chaque couche
-optimiseur
-fonction de perte


## Classe Population:
    para de la population:
        - liste des individue actuel
        - différenets listes de données pour faire de l'annaylse sur al simulation par la suite
        - seuille pour la préselection ( distance en écart type de la moyenne autoriser pour continue à être entrainé)
        - pourcentage à garder d'individu divergeant ( on garde des individu bizarre pour garder de la diversité )
        - les différents taux de mutations et amplitude de mutation
        - nombre d'individue à selectionner, muter et nombre de croisment à réaliser




# fonction : 

## création_individu:
    para entré : 
    - dico des para modifiable ( celui qui est remplie avec true ou flase)
    - type de réseau ( pour l'instant on va rester que sur des perceptrons donc la variable aura une val par défault)

    intensi l'objet 

    return : l'objet créer et initialisé


## init_population
    para entré : 
    - Nombre d'individu dans la population

    créer une liste avec tout les individu constituant la population

    return liste_population


## entrainement_indiv_begin
    para_entré: 
    - individu
    - int du nombre de batch à passer jusqu'à la première préselection des modèles


    entraîne le modèle pour les m premier batch
    remplie la liste des loss durant l'entraîenemnt

    return None


## prédiction_performance_modele
    para_entré: 
    - Individu
    
    utlise la lsite des loss pour prédire ses performance future

    return None


## premier_pre_selection
    para_entré: 
    - population

    supprime les éléments vraiment pas prometteur ( regler un paramètre de la simulation dit "seuil_preselection" qui désigne à compbien d'éacrt type de la moyen, en plus bas, doit se trouver au minimum un indiv pour pouvoir continue à être entraîné)

    return population


## entraînemnt_population
    para_entré: 
    - population

    appele la fonction entrainement_indi_begin pour chaque indivu qui n'est aps déjà entraîner
    appele la fonction qui va faire la première pre selection
    appele la second fonction d'entraînement qui va finir les train des indiv restant

    return None


## selection_dans_population
    para_entré: 
    - population

    selection les modèles par tournois, en gardant un certains pourcentage d'individus bizarres
    rajout des données opur l'analyse stockées adns certaines variables de l'objet population

    return None


## mutation_indiv
    para_entré:
    - individu
    - liste des taux de mutations et des ranges de mutations

    créer un nouvel individu qui est de base une copie de l'individu passer en paramètre puis modifie certains attributs pour effectuer les mutation

    return new_indiv



## mutation_population
    para_entré:
    - population

    appel la fonctino mutation_indiv le nombre de fois qui est en paramètes de population, à chaque fois sur un indiv prit au hasard dans la population rentré en paramètre ( on ne fait pas muté les nouveaux)
    ajoute les nouveaux individu à la population

    return None


## 