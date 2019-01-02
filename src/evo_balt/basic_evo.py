import random
from datetime import datetime

from src.evo_balt.evo import SPEA2


def basic_population(pop_size):
    random.seed(datetime.now())
    return [SPEA2.Individ(genotype=[random.randint(-10, 10), random.randint(-10, 10)]) for _ in range(pop_size)]


def basic_objectives(pop):
    for p in pop:
        obj1 = pow(p.genotype[0], 2) + pow(p.genotype[1], 2)
        obj2 = pow(p.genotype[0] - 2, 2) + pow(p.genotype[1] - 2, 2)
        p.objectives = (obj1, obj2)


def basic_mutation(individ):
    for idx in range(len(individ.genotype)):
        if random.random() > 0.8:
            sign = 1 if random.random() < 0.5 else -1
            individ.genotype[idx] += sign * 0.1

    return individ
