import os
from enum import IntEnum

import yaml


class SWANParams:
    def __init__(self, drag_func, physics_type, wcr, ws):
        '''
        Represents parameters of SWAN model that will be evolve
        :param drag_func:
        :param physics_type: GEN1, GEN3 - enum ?
        :param wcr: WhiteCappingRate
        :param ws: Wave steepness
        '''

        self.drag_func = drag_func
        self.physics_type = physics_type
        self.wcr = wcr
        self.ws = ws


class PhysicsType(IntEnum):
    GEN1 = 1
    GEN3 = 2


class EvoConfig:
    def __init__(self):
        self.content = self._load()

    def _load(self):
        with open(os.path.join(os.path.dirname(__file__), "../../evo-config.yaml"), 'r') as stream:
            return yaml.load(stream)

    def grid_by_name(self, name):
        return self.content['grid'][name]


class Evolution:
    def __init__(self):
        self.population = []

    def run(self):
        raise NotImplementedError

    def fitness(self):
        raise NotImplementedError

    def selection(self):
        raise NotImplementedError

    def mutation(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError


class SPEA2:
    def __init__(self, max_gens, pop_size, archive_size, crossover_rate):
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.crossover_rate = crossover_rate

        self._init_populations()

    def _init_populations(self):
        # TODO: init from SWANParamsFactory or smth
        self._pop = [SWANParams(1, PhysicsType.GEN3, 1, 1) for _ in range(self.pop_size)]
        self._archive = []

    def solution(self):
        gen = 0
        while True:
            # update pop, archive

            if gen >= self.max_gens:
                break

            # selection
            # mutate, crossover

            gen += 1

        return self._archive


print(SPEA2(10, 10, 10, 0.5).solution())
