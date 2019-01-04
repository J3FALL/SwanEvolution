import random

from src.swan.model import SWANParams


def calculate_objectives(model, pop):
    '''
    Calculate two error functions i.e. |model_out - observation| ^ 2
    :param model: Class that can generate SWAN-like output for a given params
    :param pop: Population of SWAN-params i.e. individuals
    '''

    for p in pop:
        params = p.genotype
        closest = model.closest_params(params)
        params.update(drag_func=closest[0], physics_type=closest[1], wcr=closest[2], ws=closest[3])
        obj_station1, obj_station2, obj_station3 = model.output(params=params)
        p.objectives = (obj_station1, obj_station2, obj_station3)


def crossover(p1, p2, rate):
    if random.random() >= rate:
        return p1

    child_params = SWANParams(drag_func=(p1.drag_func + p2.drag_func) / 2.0, physics_type=p1.physics_type,
                              wcr=(p1.wcr + p2.wcr) / 2.0, ws=p2.ws)
    return child_params


def mutation(individ):
    params = ['drag_func', 'wcr']
    if random.random() < 0.2:
        param_to_mutate = params[random.randint(0, 1)]

        sign = 1 if random.random() < 0.5 else -1
        if param_to_mutate is 'drag_func':
            individ.drag_func += sign * 0.3
        if param_to_mutate is 'wcr':
            individ.wcr += sign * 0.05
    return individ
