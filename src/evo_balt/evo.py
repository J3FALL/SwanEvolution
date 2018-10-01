import os
import random
from datetime import datetime
from math import sqrt
from operator import itemgetter

import yaml

from src.evo_balt.model import FakeModel
from src.evo_balt.model import GridFile
from src.evo_balt.model import SWANParams

random.seed(datetime.now())


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

        self.model = FakeModel(grid_file=GridFile(path="../../samples/grid_full.csv"))

    def _init_populations(self):
        self._pop = [SPEA2.Individ(genotype=SWANParams.new_instance()) for _ in range(self.pop_size)]
        # self._pop = basic_population(self.pop_size)
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
            return self.objectives[0] + self.objectives[1] + self.objectives[2]

    class ErrorHistory:
        class Point:
            def __init__(self, genotype="", genotype_index=0, fitness_value=pow(10, 9), error_value=pow(10, 9)):
                self.genotype = genotype
                self.genotype_index = genotype_index
                self.fitness_value = fitness_value
                self.error_value = error_value

        def __init__(self):
            self.history = []

        def add_new(self, genotype, genotype_index, fitness, error):
            self.history.append(SPEA2.ErrorHistory.Point(genotype=genotype, genotype_index=genotype_index,
                                                         fitness_value=fitness, error_value=error))

        def last(self):
            return SPEA2.ErrorHistory.Point() if len(self.history) == 0 else self.history[-1]

    def solution(self):
        history = SPEA2.ErrorHistory()
        gen = 0
        while gen < self.max_gens:
            self.fitness()
            self._archive = self.environmental_selection(self._pop, self._archive)
            best = sorted(self._archive, key=lambda p: p.fitness())[0]
            last_fit = history.last().fitness_value
            if last_fit > best.fitness():
                best_gens = best.genotype
                print("new best: ", best.fitness(), best.objectives,
                      sqrt(pow(best.objectives[0], 2) + pow(best.objectives[1], 2) + pow(best.objectives[2], 2)))
                print(gen)
                history.add_new(best_gens, gen, best.fitness(),
                                sqrt(pow(best.objectives[0], 2) + pow(best.objectives[1], 2) + pow(best.objectives[2],
                                                                                                   2)))
            selected = self.selected(self.pop_size, self._archive)
            self._pop = self.reproduce(selected, self.pop_size)
            gen += 1

        return history

    def fitness(self):
        self.calculate_objectives(self._pop)
        union = self._archive + self._pop
        self.calculate_dominated(union)

        for p in self._pop:
            p.raw_fitness = self.calculate_raw_fitness(p, union)
            p.density = self.calculate_density(p, union)

    def calculate_objectives(self, pop):
        '''
        Calculate two error functions i.e. |model_out - observation| ^ 2
        :param pop:
        :return:
        '''

        # Extract model_output with FakeModel for corresponding population params
        # Calculate errors

        for p in pop:
            params = p.genotype
            closest = self.model.closest_params(params)
            params.update(drag_func=closest[0], physics_type=closest[1], wcr=closest[2], ws=closest[3])
            obj_station1, obj_station2, obj_station3 = self.model.output(params=params)
            p.objectives = (obj_station1, obj_station2, obj_station3)
            # print(p.objectives, sqrt(pow(p.objectives[0], 2) + pow(p.objectives[1], 2) + pow(p.objectives[2], 2)))
        # basic_objectives(pop)

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
        if random.random() >= rate:
            return p1

        child_params = SWANParams(drag_func=p1.genotype.drag_func, physics_type=p1.genotype.physics_type,
                                  wcr=p2.genotype.wcr, ws=p2.genotype.ws)
        child = SPEA2.Individ(genotype=child_params)
        return child

    def mutation(self, individ):
        params = ['drag_func', 'wcr']
        if random.random() > 0.2:
            param_to_mutate = params[random.randint(0, 1)]

            sign = 1 if random.random() < 0.5 else -1
            if param_to_mutate is 'drag_func':
                individ.genotype.drag_func += sign * 0.1
            if param_to_mutate is 'wcr':
                individ.genotype.wcr += sign * 0.01
        return individ


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
