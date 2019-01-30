import os
import random
from functools import partial

import numpy as np

from src.evolution.spea2 import SPEA2
from src.noice_experiments.errors import (
    error_dtw_all,
    error_rmse_all
)
from src.noice_experiments.evo_operators import (
    calculate_objectives_interp,
    crossover,
    mutation,
    initial_pop_lhs,
    default_initial_pop
)
from src.noice_experiments.model import (
    CSVGridFile,
    FakeModel
)
from src.noice_experiments.model import SWANParams
from src.utils.files import (
    real_obs_from_files,
    wave_watch_results
)
from src.utils.vis import (
    plot_results,
    plot_population_movement
)

random.seed(42)

ALL_STATIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def get_rmse_for_all_stations(forecasts, observations):
    # assert len(observations) == len(forecasts) == 3

    time = np.linspace(1, 253, num=len(forecasts[0].hsig_series))

    obs_series = []
    for obs in observations:
        obs_series.append(obs.time_series(from_date="20140814.120000", to_date="20140915.000000")[:len(time)])

    results_for_stations = np.zeros(len(forecasts))
    for idx in range(0, len(forecasts)):
        results_for_stations[idx] = np.sqrt(
            np.mean((np.array(forecasts[idx].hsig_series) - np.array(obs_series[idx])) ** 2))
    return results_for_stations


def optimize_by_real_obs():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    obs = [obs.time_series(from_date="20140814.120000", to_date="20140915.000000") for obs in real_obs_from_files()]
    fake_model = FakeModel(grid_file=grid, observations=obs, stations_to_out=[1, 2, 3], error=error_dtw_all,
                           forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens=100, pop_size=10, archive_size=5,
                            crossover_rate=0.8, mutation_rate=0.6, mutation_value_rate=[0.1, 0.005, 0.005]),
        init_population=default_initial_pop,
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


def model_all_stations():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS)]

    model = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=ALL_STATIONS, error=error_rmse_all,
                      forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    return model


def default_params_forecasts(model):
    '''
    Our baseline:  forecasts with default SWAN params
    '''

    closest_params = model.closest_params(params=SWANParams(drf=1.0,
                                                            cfw=0.015,
                                                            stpm=0.00302))
    default_params = SWANParams(drf=closest_params[0], cfw=closest_params[1], stpm=closest_params[2])
    drf_idx, cfw_idx, stpm_idx = model.params_idxs(default_params)
    forecasts = model.grid[drf_idx, cfw_idx, stpm_idx]

    return forecasts


def optimize_by_ww3_obs(max_gens, pop_size, archive_size, crossover_rate, mutation_rate, mutation_value_rate):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    train_stations = [1, 2, 3]

    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=train_stations)]

    train_model = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=train_stations, error=error_rmse_all,
                            forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens, pop_size=pop_size, archive_size=archive_size,
                            crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                            mutation_value_rate=mutation_value_rate),
        init_population=initial_pop_lhs,
        objectives=partial(calculate_objectives_interp, train_model),
        crossover=crossover,
        mutation=mutation).solution(verbose=True)

    params = history.last().genotype

    forecasts = []

    test_model = model_all_stations()

    closest_hist = test_model.closest_params(params)
    closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

    for row in grid.rows:

        if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
            drf_idx, cfw_idx, stpm_idx = test_model.params_idxs(row.model_params)
            forecasts = test_model.grid[drf_idx, cfw_idx, stpm_idx]
            if grid.rows.index(row) < 100:
                print("!!!")
            print("index : %d" % grid.rows.index(row))
            break

    plot_results(forecasts=forecasts,
                 observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS),
                 baseline=default_params_forecasts(test_model))
    plot_population_movement(archive_history, grid)

    return history


def run_robustess_exp(max_gens, pop_size, archive_size, crossover_rate, mutation_rate, mutation_value_rate, stations,
                      **kwargs):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    train_model = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_all,
                            forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)
    test_model = model_all_stations()
    default_forecasts = default_params_forecasts(test_model)

    ref_metrics = get_rmse_for_all_stations(default_forecasts,
                                            wave_watch_results(path_to_results='../../samples/ww-res/',
                                                               stations=ALL_STATIONS))

    history, _ = SPEA2(
        params=SPEA2.Params(max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                            crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                            mutation_value_rate=mutation_value_rate),
        init_population=initial_pop_lhs,
        objectives=partial(calculate_objectives_interp, train_model),
        crossover=crossover,
        mutation=mutation).solution(verbose=False)

    params = history.last().genotype

    forecasts = []

    closest_hist = test_model.closest_params(params)
    closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

    for row in grid.rows:
        if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
            drf_idx, cfw_idx, stpm_idx = test_model.params_idxs(row.model_params)
            forecasts = test_model.grid[drf_idx, cfw_idx, stpm_idx]
            break

    if 'save_figures' in kwargs and kwargs['save_figures'] is True:
        plot_results(forecasts=forecasts,
                     observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS),
                     baseline=default_params_forecasts(test_model),
                     save=True, file_path=kwargs['figure_path'])

    metrics = get_rmse_for_all_stations(forecasts,
                                        wave_watch_results(path_to_results='../../samples/ww-res/',
                                                           stations=ALL_STATIONS))

    return [history.last(), metrics, ref_metrics]


objective_robustparams = {'a': 0, 'archive_size_rate': 0.3516265476722533, 'crossover_rate': 0.7194075160834003,
                          'max_gens': 3, 'mutation_p1': 0.18530572116666033, 'mutation_p2': 0.008275074614718868,
                          'mutation_p3': 0.000917588547202427, 'mutation_rate': 0.15718021655197123, 'pop_size': 19}

objective_q = {'a': 0, 'archive_size_rate': 0.18192329983957756, 'crossover_rate': 0.8275151161211388, 'max_gens': 4,
               'mutation_p1': 0.22471644990516082, 'mutation_p2': 0.004027729364749993,
               'mutation_p3': 0.000297583624177003, 'mutation_rate': 0.22663581900044313, 'pop_size': 9}

objective_tradeoff = {'a': 0, 'archive_size_rate': 0.35157832568915776, 'crossover_rate': 0.37407732045418357,
                      'max_gens': 9, 'mutation_p1': 0.21674397143802346, 'mutation_p2': 0.017216450597376923,
                      'mutation_p3': 0.0008306686136608031, 'mutation_rate': 0.2696660952766096, 'pop_size': 17}

objective_manual = {'a': 0, 'archive_size_rate': 0.3, 'crossover_rate': 0.3,
                    'max_gens': 30, 'mutation_p1': 0.1, 'mutation_p2': 0.01,
                    'mutation_p3': 0.001, 'mutation_rate': 0.5, 'pop_size': 20}


def robustness_statistics():
    papam_for_run = objective_manual

    stations_for_run_set = [[1], [2], [3], [4], [5], [6], [7], [8], [9],
                            [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9],
                            [1, 2, 3], [4, 5, 6], [7, 8, 9],
                            [1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 8, 9],
                            [1, 2, 3, 4, 5], [1, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    stations_for_run_set2 = [[1],
                             [1, 2],
                             [1, 2, 3],
                             [1, 2, 3, 4],
                             [1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5, 6],
                             [4],
                             [4, 5],
                             [4, 5, 6],
                             [4, 5, 6, 7],
                             [4, 5, 6, 7, 8],
                             [4, 5, 6, 7, 8, 9],
                             [1],
                             [1, 2],
                             [1, 3, 3],
                             [1, 1, 3, 7],
                             [1, 2, 3, 7, 8],
                             [1, 2, 3, 7, 8, 9]]
    stations_metrics = np.zeros(9)

    import datetime
    import csv

    exptime = str(datetime.datetime.now().time()).replace(":", "-")
    os.mkdir(f'../{exptime}')

    exp_id = 0
    iter_id = 0
    with open(f'../exp-res-{exptime}.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'IterId', 'SetId', 'drf', 'cfw', 'stpm',
                      'st1', 'st2', 'st3', 'st4', 'st5', 'st6', 'st7', 'st8', 'st9']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

    for _ in range(10):
        for set_id in range(0, len(stations_for_run_set2)):
            stations_for_run = stations_for_run_set2[set_id]
            archive_size = round(papam_for_run['archive_size_rate'] * papam_for_run['pop_size'])
            mutation_value_rate = [papam_for_run['mutation_p1'], papam_for_run['mutation_p2'],
                                   papam_for_run['mutation_p3']]
            best, metrics, ref_metrics = run_robustess_exp(max_gens=papam_for_run['max_gens'],
                                                           pop_size=papam_for_run['pop_size'],
                                                           archive_size=archive_size,
                                                           crossover_rate=papam_for_run['crossover_rate'],
                                                           mutation_rate=papam_for_run['mutation_rate'],
                                                           mutation_value_rate=mutation_value_rate,
                                                           stations=stations_for_run,
                                                           save_figures=True,
                                                           figure_path=os.path.join('..', exptime, str(exp_id)))

            stations_metrics[0:9] = metrics / ref_metrics
            with open(f'../exp-res-{exptime}.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'ID': exp_id, 'IterId': iter_id, 'SetId': set_id,
                                 'drf': best.genotype.drf,
                                 'cfw': best.genotype.cfw,
                                 'stpm': best.genotype.stpm,
                                 'st1': stations_metrics[0], 'st2': stations_metrics[1],
                                 'st3': stations_metrics[2],
                                 'st4': stations_metrics[3], 'st5': stations_metrics[4],
                                 'st6': stations_metrics[5],
                                 'st7': stations_metrics[6], 'st8': stations_metrics[7],
                                 'st9': stations_metrics[8]})
            exp_id += 1
        iter_id += 1

    print(stations_metrics)


# from tqdm import tqdm
#
# for _ in tqdm(range(10)):
#     robustness_statistics()
# optimize_by_ww3_obs(max_gens=150, pop_size=20, archive_size=5, crossover_rate=0.8, mutation_rate=0.7,
#                     mutation_value_rate=[0.1, 0.001, 0.0005])

robustness_statistics()
