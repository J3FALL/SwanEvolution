import random
from functools import partial

import numpy as np

from src.evolution.spea2 import SPEA2
from src.noice_experiments.errors import (
    error_rmse_peak,
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
    fake_model = FakeModel(grid_file=grid, observations=obs, stations_to_out=[1, 2, 3], error=error_rmse_peak,
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


def optimize_by_ww3_obs(max_gens, pop_size, archive_size, crossover_rate, mutation_rate, mutation_value_rate):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    stations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_all,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens, pop_size=pop_size, archive_size=archive_size,
                            crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                            mutation_value_rate=mutation_value_rate),
        init_population=default_initial_pop,
        objectives=partial(calculate_objectives_interp, fake),
        crossover=crossover,
        mutation=mutation).solution(verbose=True)

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


def run_robustess_exp(max_gens, pop_size, archive_size, crossover_rate, mutation_rate, mutation_value_rate, stations,
                      repeats):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_all,
                     forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    ww3_obs_all = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=[1, 2, 3, 4, 5, 6, 7, 8, 9])]

    fake_all = FakeModel(grid_file=grid, observations=ww3_obs_all, stations_to_out=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         error=error_rmse_all,
                         forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    closest_hist = fake.closest_params(params=SWANParams(drf=1.0,
                                                         cfw=0.015,
                                                         stpm=0.00302))
    closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

    for row in grid.rows:

        if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
            drf_idx, cfw_idx, stpm_idx = fake.params_idxs(row.model_params)
            forecasts_ref = fake_all.grid[drf_idx, cfw_idx, stpm_idx]
            print("index : %d" % grid.rows.index(row))
            break

    ref_metrics = get_rmse_for_all_stations(forecasts_ref, wave_watch_results(path_to_results='../../samples/ww-res/',
                                                                              stations=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

    obtained_params = []
    obtained_metrics = []

    all_stat_metrics = np.zeros(9)

    for t in range(1, repeats + 1):
        history, _ = SPEA2(
            params=SPEA2.Params(max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                                crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                                mutation_value_rate=mutation_value_rate),
            init_population=initial_pop_lhs,
            objectives=partial(calculate_objectives_interp, fake),
            crossover=crossover,
            mutation=mutation).solution(verbose=False)

        params = history.last().genotype

        obtained_params.append([params.drf, params.cfw, params.stpm])
        obtained_metrics.append(history.last().error_value)

        forecasts = []

        closest_hist = fake.closest_params(params)
        closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

        for row in grid.rows:

            if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
                drf_idx, cfw_idx, stpm_idx = fake.params_idxs(row.model_params)
                forecasts = fake_all.grid[drf_idx, cfw_idx, stpm_idx]
                print("index : %d" % grid.rows.index(row))
                break

                # plot_results(forecasts=forecasts,
                #             observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations),
                #             optimization_history=history)

        all_stat_metrics += get_rmse_for_all_stations(forecasts,
                                                      wave_watch_results(path_to_results='../../samples/ww-res/',
                                                                         stations=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
    all_stat_metrics = all_stat_metrics / repeats

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

    return [result_td, metrics_td, metrics_q, params_r, history.last(), all_stat_metrics, ref_metrics]


# optimize_by_real_obs()
# optimize_by_ww3_obs(28, 20, 6, 0.7, 0.7, [0.1, 0.005, 0.005])

# f = run_robustess_exp(28, 20, 6, 0.7, 0.7, [0.1, 0.005, 0.005])
# print("META FINTESS")
# print(f)
#
# f = run_robustess_exp(28, 20, 6, 0.67, 0.17)


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
stations_metrics = np.zeros(9)

import datetime
import csv

exptime = str(datetime.datetime.now().time()).replace(":", "-")

exp_id = 0
iter_id = 0
with open(f'../exp-res-{exptime}.csv', 'w', newline='') as csvfile:
    fieldnames = ['ID', 'IterId', 'SetId', 'drf', 'cfw', 'stpm',
                  'st1', 'st2', 'st3', 'st4', 'st5', 'st6', 'st7', 'st8', 'st9']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

rep_range = range(0, 10)
for rep in rep_range:
    for set_id in range(0, len(stations_for_run_set)):
        stations_for_run = stations_for_run_set[set_id]
        res = run_robustess_exp(papam_for_run['max_gens'], papam_for_run['pop_size'],
                                round(papam_for_run['archive_size_rate'] * papam_for_run['pop_size']),
                                papam_for_run['crossover_rate'],
                                papam_for_run['mutation_rate'],
                                [papam_for_run['mutation_p1'], papam_for_run['mutation_p2'],
                                 papam_for_run['mutation_p3']], stations_for_run, 1)
        best = res[4]
        ref_metrics = res[6]

        stations_metrics[0:9] = res[5] / ref_metrics
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

############
