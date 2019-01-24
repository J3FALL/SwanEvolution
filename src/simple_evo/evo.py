import os
import random
from datetime import datetime
from math import sqrt
from operator import itemgetter

import yaml

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
    def __init__(self, params, new_individ, objectives, crossover, mutation):
        '''
         Strength Pareto Evolutionary Algorithm
        :param params: Meta-parameters of the SPEA2
        :param new_individ: function to generate new individuals for population
        :param objectives: function to calculate objective functions for each individual in population
        :param crossover: function to crossover two genotypes
        :param mutation: function to mutate genotype
        '''
        self.params = params

        self.new_individ = new_individ
        self.objectives = objectives
        self.crossover = crossover
        self.mutation = mutation

        self._init_populations()

    def _init_populations(self):
        self._pop = [SPEA2.Individ(genotype=self.new_individ()) for _ in range(self.params.pop_size)]
        self._archive = []

    class Params:
        def __init__(self, max_gens, pop_size, archive_size, crossover_rate, mutation_rate):
            self.max_gens = max_gens
            self.pop_size = pop_size
            self.archive_size = archive_size
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate

    class Individ:
        def __init__(self, genotype):
            self.objectives = ()
            self.genotype = genotype
            self.dominators = []
            self.raw_fitness = 0
            self.density = 0

        def fitness(self):
            # return self.raw_fitness + self.density
            return rmse(self)

        def weighted_sum(self):
            return sum(list(self.objectives))

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
        fits = []
        history = SPEA2.ErrorHistory()
        gen = 0
        while gen < self.params.max_gens:
            self.fitness()
            self._archive = self.environmental_selection(self._pop, self._archive)
            # plot_population_movement(pop=self._pop, model=fake)

            fits.append([rmse(p) for p in
                         self._archive])

            best = sorted(self._archive, key=lambda p: p.fitness())[0]
            last_fit = history.last().fitness_value
            if last_fit > best.fitness():
                best_gens = best.genotype
                print("new best: ", round(best.fitness(),5), round(best.genotype.drf,2),round(best.genotype.cfw,6),round(best.genotype.stpm,6),
                      round(rmse(best),4))
                print(gen)
                history.add_new(best_gens, gen, best.fitness(),
                                rmse(best))
            selected = self.selected(self.params.pop_size, self._archive)
            self._pop = self.reproduce(selected, self.params.pop_size)
            gen += 1

        return history

    def fitness(self):
        self.objectives(self._pop)
        union = self._archive + self._pop
        self.calculate_dominated(union)

        for p in union:
            p.raw_fitness = self.calculate_raw_fitness(p, union)
            p.density = self.calculate_density(p, union)
            # print(p.raw_fitness, p.density, p.objectives)

            # plot_pareto(self._pop)

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

        k = int(sqrt(self.params.pop_size + self.params.archive_size))
        # k = 1
        density = 1.0 / (distances_to_src[k] + 2.0)
        return density

    def euclidean_distance(self, p1, p2):
        sum = 0
        for idx in range(len(p1)):
            sum += pow(p1[idx] - p2[idx], 2)

        return sqrt(sum)

    def environmental_selection(self, pop, archive):
        union = archive + pop
        # TODO: check value of fitness for union
        env = [p for p in union if p.fitness() < 1.0]

        if len(env) < self.params.archive_size:
            # print("adding")
            # Fill the archive with the remaining candidate solutions
            union.sort(key=lambda p: p.fitness())
            for p in union:
                if len(env) >= self.params.archive_size:
                    break
                # TODO: check value of fitness for union
                if p.fitness() >= 1.0:
                    env.append(p)
        elif len(env) > self.params.archive_size:
            while True:
                # print("truncate")
                # Truncate the archive population
                k = int(sqrt(len(env)))
                # k = 1
                dens = []
                for p1 in env:
                    distances_to_p1 = []
                    for p2 in env:
                        distances_to_p1.append(self.euclidean_distance(p1.objectives, p2.objectives))
                    distances_to_p1 = sorted(distances_to_p1)
                    # TODO: check density formula
                    density = 1.0 / (distances_to_p1[k] + 2.0)
                    dens.append((p1, density))
                dens.sort(key=itemgetter(1))
                to_remove = dens[0][0]
                env.remove(to_remove)

                if len(env) <= self.params.archive_size:
                    break
        return env

    def selected(self, size, pop):
        selected = []
        while len(selected) < size:
            selected.append(self.binary_tournament(pop))

        return selected

    def binary_tournament(self, pop):
        # random.seed(datetime.now())

        i, j = random.randint(0, len(pop) - 1), random.randint(0, len(pop) - 1)

        while j == i:
            j = random.randint(0, len(pop) - 1)
        return pop[i] if pop[i].fitness() < pop[j].fitness() else pop[j]

    def reproduce(self, selected, pop_size):
        children = []
        # selected = sorted(selected, key=lambda p: p.fitness())
        #
        # best_range = int(0.3 * pop_size)
        # best_parents = selected[:best_range]
        # children.extend(best_parents)
        for p1 in selected:
            idx = selected.index(p1)
            p2 = selected[idx + 1] if idx % 2 == 0 else selected[idx - 1]
            if idx == len(selected) - 1:
                p2 = selected[0]

            child_gen = self.crossover(p1.genotype, p2.genotype, self.params.crossover_rate)
            child_gen = self.mutation(child_gen, self.params.mutation_rate)
            child = SPEA2.Individ(genotype=child_gen)
            children.append(child)

            if len(children) >= pop_size:
                break

        return children


def rmse(individ):
    result = 0.0
    for obj in individ.objectives:
        result += pow(obj, 2)
    return sqrt(result / len(individ.objectives))
