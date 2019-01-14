import csv
from functools import partial
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from src.noice_experiments.evo_operators import (
    calculate_objectives,
    crossover,
    mutation
)
from src.noice_experiments.model import (
    CSVGridFile,
    FakeModel
)
from src.noice_experiments.model import SWANParams
from src.simple_evo.evo import SPEA2
from src.swan.files import ObservationFile

grid = CSVGridFile('../../samples/wind-exp-params.csv')

fake = FakeModel(grid_file=grid, forecasts_path='../../samples/wind-noice-runs/results/1/')


def optimize():
    history = SPEA2(
        params=SPEA2.Params(max_gens=1000, pop_size=10, archive_size=5, crossover_rate=0.8, mutation_rate=0.8),
        new_individ=SWANParams.new_instance,
        objectives=partial(calculate_objectives, fake),
        crossover=crossover,
        mutation=mutation).solution()

    params = history.last().genotype

    forecasts = []
    for row in grid.rows:
        if set(row.model_params.params_list()) == set(params.params_list()):
            drf_idx, cfw_idx, stpm_idx = fake.params_idxs(row.model_params)
            forecasts = fake.grid[drf_idx, cfw_idx, stpm_idx]
            print("index : %d" % grid.rows.index(row))

    waves_1 = ObservationFile(path="../../samples/obs/1a_waves.txt").time_series(from_date="20140814.120000",
                                                                                 to_date="20140915.000000")
    waves_2 = ObservationFile(path="../../samples/obs/2a_waves.txt").time_series(from_date="20140814.120000",
                                                                                 to_date="20140915.000000")
    waves_3 = ObservationFile(path="../../samples/obs/3a_waves.txt").time_series(from_date="20140814.120000",
                                                                                 to_date="20140915.000000")

    fig, axs = plt.subplots(2, 2)

    time = np.linspace(1, 253, num=len(forecasts[0].hsig_series))
    axs[0, 0].plot(time, waves_1, label='Observations, Station 1')
    axs[0, 0].plot(time, forecasts[0].hsig_series, label='Predicted, Station 1')
    axs[0, 0].legend()
    axs[0, 1].plot(time, waves_2, label='Observations, Station 2')
    axs[0, 1].plot(time, forecasts[1].hsig_series, label='Predicted, Station 2')
    axs[0, 1].legend()
    axs[1, 0].plot(time, waves_3, label='Observations, Station 3')
    axs[1, 0].plot(time, forecasts[2].hsig_series, label='Predicted, Station 3')
    axs[1, 0].legend()

    gens = [error.genotype_index for error in history.history]
    error_vals = [error.error_value for error in history.history]

    axs[1, 1].plot()
    axs[1, 1].plot(gens, error_vals, label='Loss history', marker=".")
    axs[1, 1].legend()

    plt.show()

    return history


def grid_rmse():
    errors_total = []
    with open('params_rmse.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['ID', 'DRF', 'CFW', 'STPM', 'RMSE_K1', 'RMSE_K2', 'RMSE_K3', 'TOTAL_RMSE'])
        for row in grid.rows:
            error = fake.output(params=row.model_params)
            errors_total.append(rmse(error))

            row_to_write = row.model_params.params_list()
            row_to_write.extend(error)
            row_to_write.append(rmse(error))
            writer.writerow(row_to_write)

    print(f'min total rmse: {min(errors_total)}')


def rmse(vars):
    return sqrt(sum([pow(v, 2) for v in vars]) / len(vars))


optimize()
