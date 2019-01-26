import csv
from functools import partial
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from src.noice_experiments.errors import (
    error_rmse_peak,
    error_rmse_all
)
from src.noice_experiments.evo_operators import (
    calculate_objectives_interp,
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
    real_obs_from_files,
    wave_watch_results
)


def real_obs_config():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    obs = [obs.time_series(from_date="20140814.120000", to_date="20140915.000000") for obs in real_obs_from_files()]
    fake = FakeModel(grid_file=grid, observations=obs, stations_to_out=[1, 2, 3], error=error_rmse_peak,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    return fake, grid


def optimize_by_real_obs():
    fake_model, grid = real_obs_config()

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens=50, pop_size=10, archive_size=5, crossover_rate=0.8, mutation_rate=0.8),
        new_individ=SWANParams.new_instance,
        objectives=partial(calculate_objectives_interp, fake_model),
        crossover=crossover,
        mutation=mutation).solution()

    plot_population_movement(archive_history, grid)
    params = history.last().genotype

    forecasts = []
    for row in grid.rows:
        if set(row.model_params.params_list()) == set(params.params_list()):
            drf_idx, cfw_idx, stpm_idx = fake_model.params_idxs(row.model_params)
            forecasts = fake_model.grid[drf_idx, cfw_idx, stpm_idx]

    observations = real_obs_from_files()
    plot_results(forecasts=forecasts, observations=observations)
    return history


def optimize_by_ww3_obs():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    stations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_all,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens=20, pop_size=10, archive_size=5, crossover_rate=0.8, mutation_rate=0.8),
        new_individ=SWANParams.new_instance,
        objectives=partial(calculate_objectives_interp, fake),
        crossover=crossover,
        mutation=mutation).solution()

    params = history.last().genotype

    forecasts = []

    closest_hist = fake.closest_params(params)
    closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

    for row in grid.rows:

        if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
            drf_idx, cfw_idx, stpm_idx = fake.params_idxs(row.model_params)
            forecasts = fake.grid[drf_idx, cfw_idx, stpm_idx]
            if grid.rows.index(row) < 100:
                print("!!!")
            print("index : %d" % grid.rows.index(row))
            break

    plot_results(forecasts=forecasts,
                 observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations))
    plot_population_movement(archive_history, grid)

    return history


def run_robustess_exp(max_gens, pop_size, archive_size, crossover_rate, mutation_rate):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    import random
    random.seed(42)

    stations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_all,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    obtained_params = []
    obtained_metrics = []
    for t in range(1, 10):
        history, _ = SPEA2(
            params=SPEA2.Params(max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                                crossover_rate=crossover_rate, mutation_rate=mutation_rate),
            new_individ=SWANParams.new_instance,
            objectives=partial(calculate_objectives_interp, fake),
            crossover=crossover,
            mutation=mutation).solution()

        params = history.last().genotype

        obtained_params.append([params.drf, params.cfw, params.stpm])
        obtained_metrics.append(history.last().error_value)

        forecasts = []

        closest_hist = fake.closest_params(params)
        closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

        for row in grid.rows:

            if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
                drf_idx, cfw_idx, stpm_idx = fake.params_idxs(row.model_params)
                forecasts = fake.grid[drf_idx, cfw_idx, stpm_idx]
                print("index : %d" % grid.rows.index(row))
                break

    # plot_results(forecasts=forecasts,
    #              observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations),
    #              optimization_history=history)

    print("ROBUSTNESS METRICS")
    print("DRAG SD, %")
    drag_sdm = np.std([i[0] for i in obtained_params]) / 1 * 100
    print(round(drag_sdm, 4))
    print("CFW SD, %")
    cfw_sdm = np.std([i[1] for i in obtained_params]) / 0.015 * 100
    print(round(cfw_sdm, 4))
    print("STMP SD, %")
    stpm_sdm = np.std([i[2] for i in obtained_params]) / 0.00302 * 100
    print(round(stpm_sdm, 4))
    print("FITNESS SD, %")
    print(round(np.std(obtained_metrics) / np.mean(obtained_metrics) * 100, 4))

    print("QUALITY METRICS")
    print("MEAN")
    print(round(np.mean(obtained_metrics), 2))
    print("MAX")
    print(round(np.max(obtained_metrics), 2))
    print("MIN")
    print(round(np.min(obtained_metrics), 2))
    print("PARAMS")
    print(max_gens, pop_size, archive_size, crossover_rate, mutation_rate)

    result_td = np.mean(obtained_metrics) * (np.std(obtained_metrics) / np.mean(obtained_metrics) * 100)

    metrics_td = np.mean(obtained_metrics) * (drag_sdm * cfw_sdm * stpm_sdm)

    metrics_q = np.mean(obtained_metrics)

    params_r = (drag_sdm * cfw_sdm * stpm_sdm)

    return [result_td, metrics_td, metrics_q, params_r]


def plot_results(forecasts, observations):
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

    plt.show()


def plot_population_movement(archive_history, grid):
    fig = plt.figure()
    ax = Axes3D(fig)
    for pop in archive_history:

        drf = [individ.genotype.drf for individ in pop]
        cfw = [individ.genotype.cfw for individ in pop]
        stpm = [individ.genotype.stpm for individ in pop]

        pop_idx = archive_history.index(pop)

        rmse_values = []
        max_idx = -1
        for point_idx in range(len(pop)):
            rmse_val = rmse([obj for obj in pop[point_idx].objectives])
            rmse_values.append(rmse_val)
            max_idx = rmse_values.index(max(rmse_values))
        for point_idx in range(len(pop)):
            color = 'red' if point_idx is not max_idx else 'blue'
            ax.scatter(drf[point_idx], cfw[point_idx], stpm[point_idx], c=color, s=5)
            ax.text(drf[point_idx], cfw[point_idx], stpm[point_idx], f'{rmse_val:.2f}({pop_idx})', size=10, zorder=1,
                    color='k')

    ax.set_xlim(left=min(grid.drf_grid), right=max(grid.drf_grid))
    ax.set_ylim(bottom=min(grid.cfw_grid), top=max(grid.cfw_grid))
    ax.set_zlim(bottom=min(grid.stpm_grid), top=max(grid.stpm_grid))
    ax.set_xlabel('drf')
    ax.set_ylabel('cfw')
    ax.set_zlabel('stpm')
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


# optimize_by_real_obs()
# optimize_by_ww3_obs()

f = run_robustess_exp(7, 10, 6, 0.29, 0.6)
print("META FINTESS")
print(f)

f = run_robustess_exp(28, 20, 6, 0.67, 0.17)
