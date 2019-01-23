import csv
from functools import partial
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from src.noice_experiments.errors import (
    error_rmse_peak
)
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
from src.swan.files import (
    real_obs_from_files
)
from src.ww3.files import wave_watch_results


def real_obs_config():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    obs = [obs.time_series(from_date="20140814.120000", to_date="20140915.000000") for obs in real_obs_from_files()]
    fake = FakeModel(grid_file=grid, observations=obs, stations_to_out=[1, 2, 3], error=error_rmse_peak,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    return fake, grid


def optimize_by_real_obs():
    fake_model, grid = real_obs_config()

    history = SPEA2(
        params=SPEA2.Params(max_gens=1000, pop_size=10, archive_size=5, crossover_rate=0.8, mutation_rate=0.8),
        new_individ=SWANParams.new_instance,
        objectives=partial(calculate_objectives, fake_model),
        crossover=crossover,
        mutation=mutation).solution()

    params = history.last().genotype

    forecasts = []
    for row in grid.rows:
        if set(row.model_params.params_list()) == set(params.params_list()):
            drf_idx, cfw_idx, stpm_idx = fake_model.params_idxs(row.model_params)
            forecasts = fake_model.grid[drf_idx, cfw_idx, stpm_idx]

    observations = real_obs_from_files()
    plot_results(forecasts=forecasts, observations=observations, optimization_history=history)
    return history


def optimize_by_ww3_obs():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    stations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_peak,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    history = SPEA2(
        params=SPEA2.Params(max_gens=100, pop_size=50, archive_size=5, crossover_rate=0.8, mutation_rate=0.8),
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

    plot_results(forecasts=forecasts,
                 observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations),
                 optimization_history=history)

    return history


def plot_results(forecasts, observations, optimization_history):
    # assert len(observations) == len(forecasts) == 3

    fig, axs = plt.subplots(3, 3)
    time = np.linspace(1, 253, num=len(forecasts[0].hsig_series))

    obs_series = []
    for obs in observations:
        obs_series.append(obs.time_series(from_date="20140814.120000", to_date="20140915.000000")[:len(time)])
    for idx in range(len(forecasts)):
        i, j = divmod(idx, 3)
        station_idx = observations[idx].station_idx
        axs[i, j].plot(time, obs_series[idx],
                       label=f'Observations, Station {station_idx}')
        axs[i, j].plot(time, forecasts[idx].hsig_series,
                       label=f'Predicted, Station {station_idx}')
        axs[i, j].legend()

    gens = [error.genotype_index for error in optimization_history.history]
    error_vals = [error.error_value for error in optimization_history.history]

    # axs[1, 1].plot()
    # axs[1, 1].plot(gens, error_vals, label='Loss history', marker=".")
    # axs[1, 1].legend()

    plt.show()


def grid_rmse():
    # fake, grid = real_obs_config()

    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    stations = [1, 2, 3]

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_peak,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    errors_total = []
    m_error = pow(10, 9)
    with open('../../samples/params_rmse.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['ID', 'DRF', 'CFW', 'STPM', 'RMSE_K1', 'RMSE_K2', 'RMSE_K3', 'TOTAL_RMSE'])
        for row in grid.rows:
            error = fake.output(params=row.model_params)
            print(grid.rows.index(row), error)

            if m_error > rmse(error):
                m_error = rmse(error)
                print(f"new min: {m_error}; {row.model_params.params_list()}")
            errors_total.append(rmse(error))

            row_to_write = row.model_params.params_list()
            row_to_write.extend(error)
            row_to_write.append(rmse(error))
            writer.writerow(row_to_write)

    print(f'min total rmse: {min(errors_total)}')


def rmse(vars):
    return sqrt(sum([pow(v, 2) for v in vars]) / len(vars))


optimize_by_ww3_obs()
# optimize_by_real_obs()

# grid_rmse()
