from src.algorithm.benchmarks import schaffer
from src.algorithm.spea2 import SPEA2


class SPEA2Test:
    def test_schaffer_function_optimization(self):
        alg = SPEA2(
            params=SPEA2.Params(max_gens=200, pop_size=10, archive_size=10, crossover_rate=0.6, mutation_rate=0.25),
            new_individ=schaffer.new_individ,
            objectives=schaffer.objectives,
            crossover=schaffer.crossover,
            mutation=schaffer.mutation,
            pop_variance=schaffer.objectives_sum)

        history = alg.solution()
        last_variance = history.last().pop_variance

        assert is_less_than(last_variance, 10)


def is_less_than(values, limit):
    greater_than = [v for v in values if v > limit]
    print(greater_than)
    return len(greater_than) == 0
