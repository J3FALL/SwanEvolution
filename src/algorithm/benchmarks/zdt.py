# Zitzler–Deb–Thiele's function N. 1

import random
from datetime import datetime
from math import sqrt

PROBLEM_SIZE = 30


def new_individ():
    random.seed(datetime.now())
    return [random.uniform(0, 1) for _ in range(PROBLEM_SIZE)]


def objectives(pop):
    for p in pop:
        f1 = p.genotype[0]
        g = 1.0 + 9.0 * sum(p.genotype[1:]) / (PROBLEM_SIZE - 1)
        h = 1.0 - sqrt(f1 / g)
        f2 = g * h
        p.objectives = (f1, f2)


def mutation(genotype, rate):
    for idx in range(len(genotype)):
        if random.random() < rate:
            sign = 1 if random.random() < 0.5 else -1
            genotype[idx] += sign * 0.1
            genotype[idx] = max(0, genotype[idx])
            genotype[idx] = min(1, genotype[idx])
    return genotype


def crossover(gen1, gen2, rate):
    if random.random() > rate:
        return gen1

    child_gen = [(gen1[idx] + gen2[idx]) / 2.0 for idx in range(len(gen1))]
    return child_gen
