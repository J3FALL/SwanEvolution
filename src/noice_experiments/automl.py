from hyperopt import fmin, tpe, space_eval
from hyperopt import hp

from src.noice_experiments.main import run_robustess_exp

space = hp.choice('a',
                  [
                      (hp.randint("max_gens", 15) * 2 + 6, hp.randint("pop_size", 20) + 10,
                       hp.uniform("archive_size_rate", 0.05, 0.4) * 2, hp.uniform("crossover_rate", 0.1, 0.9),
                       hp.uniform("mutation_rate", 0.1, 0.9))
                  ])

score_history = []
best_score_history = []

best_score = 99999


def objective(args):
    max_gens, pop_size, archive_size_rate, crossover_rate, mutation_rate = args
    print(max_gens, pop_size, archive_size_rate, crossover_rate, mutation_rate)
    archive_size = round(archive_size_rate * pop_size)
    score = run_robustess_exp(max_gens, pop_size, archive_size, crossover_rate, mutation_rate)

    if (len(best_score_history) == 0 or score < best_score_history[-1]):
        best_score = score
        best_score_history.append(best_score)
    score_history.append(score)

    return score[1]


print("START OPT")
best = fmin(objective, space, algo=tpe.suggest, max_evals=30, verbose=True)
print("END OPT")
print(best)
print(space_eval(space, best))
