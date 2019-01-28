from functools import partial

from src.ensemble.ensemble import Ensemble
from src.evolution.spea2 import SPEA2
from src.noice_experiments.errors import error_rmse_all
from src.noice_experiments.evo_operators import (
    calculate_objectives_interp,
    crossover,
    mutation,
    initial_pop_lhs
)
from src.noice_experiments.model import (
    CSVGridFile,
    SWANParams,
    FakeModel
)
from src.utils.files import (
    real_obs_from_files,
    wave_watch_results)
from src.utils.vis import plot_results, plot_population_movement


def optimize():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    stations = [1, 2, 3]

    obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    # obs = [obs.time_series(from_date="20140814.120000", to_date="20140915.000000") for obs in
    #        real_obs_from_files()]

    base_model = FakeModel(grid_file=grid, observations=obs, stations_to_out=stations, error=error_rmse_all,
                           forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    ens = Ensemble(grid=grid, noise_cases=[1, 15, 25, 26], observations=obs,
                   path_to_forecasts='../../../wind-noice-runs/results_fixed',
                   stations_to_out=stations, error=error_rmse_all)

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens=30, pop_size=10, archive_size=2,
                            crossover_rate=0.8, mutation_rate=0.8, mutation_value_rate=[0.1, 0.005, 0.005]),
        init_population=initial_pop_lhs,
        objectives=partial(calculate_objectives_interp, ens),
        crossover=crossover,
        mutation=mutation).solution()

    params = history.last().genotype

    closest_hist = base_model.closest_params(params)
    closest_params_set_hist = SWANParams(drf=closest_hist[0], cfw=closest_hist[1], stpm=closest_hist[2])

    forecasts = []
    for row in grid.rows:
        if set(row.model_params.params_list()) == set(closest_params_set_hist.params_list()):
            drf_idx, cfw_idx, stpm_idx = base_model.params_idxs(row.model_params)
            forecasts = base_model.grid[drf_idx, cfw_idx, stpm_idx]

    print(params.params_list())
    print(base_model.output(params=params))
    observations = real_obs_from_files()
    plot_results(forecasts=forecasts, observations=observations)

    plot_population_movement(archive_history, grid)


    return history


optimize()
