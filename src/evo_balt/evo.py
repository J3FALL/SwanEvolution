import os
import random
from datetime import datetime
from enum import IntEnum
from math import sqrt
from operator import itemgetter

import yaml

random.seed(datetime.now())


class SWANParams:
    @staticmethod
    def new_instance():
        return SWANParams(drag_func=random.uniform(0, 5), physics_type=PhysicsType.GEN3,
                          wcr=random.uniform(pow(10, -10), 2), ws=0.00302)

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
    GEN1 = 0
    GEN3 = 1


class EvoConfig:
    def __init__(self):
        self.content = self._load()

    def _load(self):
        with open(os.path.join(os.path.dirname(__file__), "../../evo-config.yaml"), 'r') as stream:
            return yaml.load(stream)

    def grid_by_name(self, name):
        return self.content['grid'][name]


class SPEA2:
    def __init__(self, max_gens, pop_size, archive_size, crossover_rate):
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.crossover_rate = crossover_rate

        self._init_populations()

    def _init_populations(self):
        # self._pop = [SPEA2.Individ(genotype=SWANParams.new_instance()) for _ in range(self.pop_size)]
        self._pop = basic_population(self.pop_size)
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

        def weighted_sum(self):
            return self.objectives[0] + self.objectives[1]

    def solution(self):
        gen = 0
        best_all = pow(10, 9)
        while True:
            self.fitness()
            self._archive = self.environmental_selection(self._pop, self._archive)
            best = sorted(self._archive, key=lambda p: p.fitness())[0]

            if best.fitness() < best_all:
                best_all = best.fitness()
                best_gens = best.genotype
                print("new best: ", best_gens, best.fitness(), best.objectives)
                print(gen)
            if gen >= self.max_gens:
                break

            selected = self.selected(self.pop_size, self._archive)
            self._pop = self.reproduce(selected, self.pop_size)

            gen += 1

        return self._archive

    def fitness(self):
        self.calculate_objectives(self._pop)
        union = self._archive + self._pop
        self.calculate_dominated(union)

        for p in self._pop:
            p.raw_fitness = self.calculate_raw_fitness(p, union)
            p.density = self.calculate_density(p, union)
            # print(p.fitness())

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
            obj1 = pow(p.genotype[0], 2) + pow(p.genotype[1], 2)
            obj2 = pow(p.genotype[0] - 2, 2) + pow(p.genotype[1] - 2, 2)
            p.objectives = (obj1, obj2)

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

    def environmental_selection(self, pop, archive):
        union = archive + pop
        env = [p for p in union if p.fitness() < 1.0]

        if len(env) < self.archive_size:
            # Fill the archive with the remaining candidate solutions
            union.sort(key=lambda p: p.fitness())
            for p in union:
                if len(env) >= self.archive_size:
                    break
                if p.fitness() >= 1.0:
                    env.append(p)
        elif len(env) > self.archive_size:
            while True:
                # Truncate the archive population
                k = int(sqrt(len(env)))

                dens = []
                for p1 in env:
                    distances_to_p1 = []
                    for p2 in env:
                        distances_to_p1.append(self.euclidean_distance(p1.objectives, p2.objectives))
                    distances_to_p1 = sorted(distances_to_p1)
                    density = 1.0 / (distances_to_p1[k] + 2.0)
                    dens.append((p1, density))
                dens.sort(key=itemgetter(1))
                to_remove = dens[0][0]
                env.remove(to_remove)

                if len(env) <= self.archive_size:
                    break
        return env

    def selected(self, size, pop):
        selected = []
        while len(selected) < size:
            selected.append(self.binary_tournament(pop))

        return selected

    def binary_tournament(self, pop):
        random.seed(datetime.now())

        i, j = random.randint(0, len(pop) - 1), random.randint(0, len(pop) - 1)

        while j == i:
            j = random.randint(0, len(pop) - 1)

        return pop[i] if pop[i].fitness() < pop[j].fitness() else pop[j]

    def reproduce(self, selected, pop_size):
        children = []

        for p1 in selected:
            idx = selected.index(p1)
            p2 = selected[idx + 1] if idx % 2 == 0 else selected[idx - 1]
            if idx == len(selected) - 1:
                p2 = selected[0]

            child = self.crossover(p1, p2, self.crossover_rate)
            child = self.mutation(child)
            children.append(child)

            if len(children) >= pop_size:
                break

        return children

    def crossover(self, p1, p2, rate):
        # TODO: crossover swan params
        if random.random() >= rate:
            return p1

        child = SPEA2.Individ(genotype=[p1.genotype[0], p2.genotype[1]])
        return child

    def mutation(self, individ):
        # TODO: mutation swan params
        for idx in range(len(individ.genotype)):
            if random.random() > 0.5:
                sign = 1 if random.random() < 0.5 else -1
                individ.genotype[idx] += sign * 0.1

        return individ


def basic_population(pop_size):
    random.seed(datetime.now())
    return [SPEA2.Individ(genotype=[random.randint(-10, 10), random.randint(-10, 10)]) for _ in range(pop_size)]


# print(SPEA2(1000, 50, 30, 0.9).solution())
