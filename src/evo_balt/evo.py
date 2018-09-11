from enum import Enum


class SWANIndivid:
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


class PhysicsType(Enum):
    GEN1 = 1
    GEN3 = 2


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
