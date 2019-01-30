import csv
from math import sqrt

from src.noice_experiments.errors import (
    error_rmse_peak,
    error_rmse_all
)
from src.noice_experiments.main import get_rmse_for_all_stations
from src.noice_experiments.model import (
    CSVGridFile,
    FakeModel
)
from src.utils.files import (
    wave_watch_results
)


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


def error_grid(noise_case=0):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    stations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ww3_obs_all = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    model_all = FakeModel(grid_file=grid, observations=ww3_obs_all, stations_to_out=stations,
                          error=error_rmse_all,
                          forecasts_path=f'../../../wind-noice-runs/results_fixed/{noise_case}', noise_run=noise_case)

    with open(f'../../samples/params_rmse_{noise_case}.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        header = ['DRF', 'CFW', 'STPM']
        error_columns = [f'ERROR_K{station}' for station in stations]
        header.extend(error_columns)
        writer.writerow(header)

        for row in grid.rows:
            drf_idx, cfw_idx, stpm_idx = model_all.params_idxs(row.model_params)
            forecasts = model_all.grid[drf_idx, cfw_idx, stpm_idx]
            metrics = get_rmse_for_all_stations(forecasts, wave_watch_results(path_to_results='../../samples/ww-res/',
                                                                              stations=stations))

            row_to_write = row.model_params.params_list()
            row_to_write.extend(metrics)

            writer.writerow(row_to_write)


def all_error_grids():
    for noise_case in [0, 1, 2, 15, 16, 17, 25, 26]:
        error_grid(noise_case)
