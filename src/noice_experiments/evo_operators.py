import random

import numpy as np

from src.noice_experiments.model import SWANParams


def calculate_objectives(model, pop):
    '''
    Calculate two error functions i.e. |model_out - observation| ^ 2
    :param model: Class that can generate SWAN-like output for a given params
    :param pop: Population of SWAN-params i.e. individuals
    '''

    for p in pop:
        params = p.genotype
        closest = model.closest_params(params)
        params.update(drf=closest[0], cfw=closest[1], stpm=closest[2])
        p.objectives = tuple(model.output(params=params))


def calculate_objectives_interp(model, pop):
    '''
    Calculate two error functions i.e. |model_out - observation| ^ 2
    :param model: Class that can generate SWAN-like output for a given params
    :param pop: Population of SWAN-params i.e. individuals
    '''

    for p in pop:
        params = p.genotype
        p.objectives = tuple(model.output(params=params))


def crossover(p1, p2, rate):
    if random.random() >= rate:
        return p1

    params = ['drf', 'cfw', 'stpm']
    param_to_mutate = params[random.randint(0, 2)]

    drf = p1.drf
    cfw = p1.cfw
    stpm = p1.stpm

    if param_to_mutate is 'drf':
        drf = p2.drf
    if param_to_mutate is 'cfw':
        cfw = p2.cfw
    if param_to_mutate is 'stpm':
        stpm = p2.stpm

    child_params = SWANParams(drf=drf,
                              cfw=cfw,
                              stpm=stpm)
    return child_params


def mutation(individ, rate, mutation_value_rate):
    params = ['drf', 'cfw', 'stpm']
    if random.random() >= rate:
        param_to_mutate = params[random.randint(0, 2)]
        mutation_ratio = abs(np.random.normal(0, 5, 1)[0])

        sign = 1 if random.random() < 0.5 else -1
        if param_to_mutate is 'drf':
            individ.drf += sign * mutation_value_rate[0] * mutation_ratio
            individ.drf = abs(individ.drf)
        if param_to_mutate is 'cfw':
            individ.cfw += sign * mutation_value_rate[1] * mutation_ratio
            individ.cfw = abs(individ.cfw)
        if param_to_mutate is 'stpm':
            individ.stpm += sign * mutation_value_rate[2] * mutation_ratio
            individ.stpm = abs(individ.stpm)
    return individ
