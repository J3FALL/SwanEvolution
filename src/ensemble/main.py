from functools import partial

from src.ensemble.ensemble import Ensemble
from src.evolution.spea2 import SPEA2
from src.noice_experiments.errors import error_rmse_peak
from src.noice_experiments.evo_operators import (
    calculate_objectives,
    crossover,
    mutation
)
from src.noice_experiments.model import (
    CSVGridFile,
    SWANParams,
    FakeModel
)
from src.utils.files import real_obs_from_files
from src.utils.vis import plot_results


def optimize():
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    obs = [obs.time_series(from_date="20140814.120000", to_date="20140915.000000") for obs in
           real_obs_from_files()]

    base_model = FakeModel(grid_file=grid, observations=obs, stations_to_out=[1, 2, 3], error=error_rmse_peak,
                           forecasts_path='../../../wind-noice-runs/results_fixed/0', noise_run=0)

    ens = Ensemble(grid=grid, noise_cases=[1, 15, 25, 26], observations=obs,
                   path_to_forecasts='../../../wind-noice-runs/results_fixed',
                   stations_to_out=[1, 2, 3])

    history, _ = SPEA2(
        params=SPEA2.Params(max_gens=150, pop_size=10, archive_size=5, crossover_rate=0.8, mutation_rate=0.8),
        new_individ=SWANParams.new_instance,
        objectives=partial(calculate_objectives, ens),
        crossover=crossover,
        mutation=mutation).solution()

    params = history.last().genotype

    forecasts = []
    for row in grid.rows:
        if set(row.model_params.params_list()) == set(params.params_list()):
            drf_idx, cfw_idx, stpm_idx = base_model.params_idxs(row.model_params)
            forecasts = base_model.grid[drf_idx, cfw_idx, stpm_idx]

    drf, cfw, stpm = base_model.closest_params(params)
    print(params.params_list())
    print(base_model.output(params=SWANParams(drf=drf, cfw=cfw, stpm=stpm)))
    observations = real_obs_from_files()
    plot_results(forecasts=forecasts, observations=observations)

    return history


optimize()
