import random
from datetime import datetime


def new_individ():
    random.seed(datetime.now())
    return [random.randint(-100, 100)]


def objectives_sum(population):
    return [p.objectives[0] + p.objectives[1] for p in population]


def objectives(pop):
    for p in pop:
        obj1 = pow(p.genotype[0], 2)
        obj2 = pow(p.genotype[0] - 2, 2)
        p.objectives = (obj1, obj2)


def mutation(genotype, rate):
    for idx in range(len(genotype)):
        if random.random() < rate:
            sign = 1 if random.random() < 0.5 else -1
            genotype[idx] += sign * 5

    return genotype


def crossover(gen1, gen2, rate):
    if random.random() > rate:
        return gen1

    child_gen = [(gen1[0] + gen2[0]) / 2.0]
    return child_gen
