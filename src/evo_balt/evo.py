import os
from enum import IntEnum
from math import sqrt

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
        self._pop = [SPEA2.Individ(SWANParams(1, PhysicsType.GEN3, 1, 1)) for _ in range(self.pop_size)]
        self._archive = []

    class Individ:
        def __init__(self, genotype):
            self.objectives = ()
            self.genotype = genotype
            self.dominators = []
            self.raw_fitness = 0
            self.density = 0

        def fitness(self):
            return self.raw_fitness + self.density

    def solution(self):
        gen = 0
        while True:
            self.fitness()
            if gen >= self.max_gens:
                break

            # selection
            # mutate, crossover

            gen += 1

        return self._archive

    def fitness(self):
        self.calculate_objectives(self._pop)
        union = self._archive + self._pop
        self.calculate_dominated(union)

        for p in self._pop:
            p.raw_fitness = self.calculate_raw_fitness(p, union)
            p.density = self.calculate_density(p, union)
            print(p.fitness())

    def calculate_objectives(self, pop):
        '''
        Calculate two error functions i.e. |model_out - observation| ^ 2
        :param pop:
        :return:
        '''

        # Extract model_output with FakeModel for corresponding population params
        # Calculate errors

        for p in pop:
            # p.objectives = (p.genotype.drag_func - 0.5, p.genotype.drag_func - 0.3)
            p.objectives = (pop.index(p), pop.index(p))

    def calculate_dominated(self, pop):
        '''
        For each individ find all individuals that are better than selected
        :return:
        '''

        for p in pop:
            p.dominators = [selected for selected in pop if selected != p and self.dominates(p, selected)]

    def dominates(self, src, target):
        for idx in range(len(src.objectives)):
            if src.objectives[idx] > target.objectives[idx]:
                return False
        return True

    def calculate_raw_fitness(self, src, pop):
        sum = 0
        for p in pop:
            if self.dominates(p, src):
                sum += len(p.dominators)
        return sum

    def calculate_density(self, src, pop):
        '''
        Estimate the density of Pareto front given k-nearest neighbour of src
        :param src:
        :param pop:
        :return:
        '''
        distances_to_src = []
        for p in pop:
            distances_to_src.append(self.euclidean_distance(src.objectives, p.objectives))
        distances_to_src = sorted(distances_to_src)

        k = int(sqrt(self.pop_size))
        density = 1.0 / (distances_to_src[k] + 2.0)
        return density

    def euclidean_distance(self, p1, p2):
        sum = 0
        for idx in range(len(p1)):
            sum += pow(p1[idx] - p2[idx], 2)

        return sqrt(sum)


print(SPEA2(10, 10, 10, 0.5).solution())
