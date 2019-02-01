from src.noice_experiments.evo_operators import (
    initial_pop_lhs_from_file
)

pop = initial_pop_lhs_from_file('pop.pik')

for p in pop:
    print(p.params_list())
