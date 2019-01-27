from src.noice_experiments.evo_operators import initial_population_lhs

pop = initial_population_lhs(10)

for p in pop:
    print(p.params_list())
