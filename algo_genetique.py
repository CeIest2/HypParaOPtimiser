

import random
import objet_pour_AG as AG
import mutation as muta
import croisement as cr
import train as tr
import selection as sel
import time


def algo_genetique(nb_generation=5) -> None:
    start = time.time()
    liste_genome_par_generation = []
    population = AG.Population(10)
    for __ in range(nb_generation):
        print("############# new generatino ###############")
        tr.train_population(population)
        # on récupère la liste des génomes pour pouvoir faire de l'ananylse dessu après coup
        liste_genome_par_generation.append([indiv.genome for indiv in population.liste_indiv])

        population = sel.selection(population, 7)
        tr.train_population(population, 1)
        population = sel.selection(population, 5)
        for _ in range(3):
            population.liste_indiv.append(muta.mutation(population.liste_indiv[random.randint(0, len(population.liste_indiv) - 1)]))
        for _ in range(2):
            population.liste_indiv.append(cr.croisement(population.liste_indiv[random.randint(0, len(population.liste_indiv) - 1)], population.liste_indiv[random.randint(0, len(population.liste_indiv) - 1)], population.param_AG))
        print(f"La nouvelle population a {len(population.liste_indiv)} individus")
        print(f" les ecart type de fin sont {[indiv.accu_final_et for indiv in population.liste_indiv]}")
        population = population
    print(f"Temps d'execution {time.time() - start}")