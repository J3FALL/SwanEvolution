import os

import numpy as np

from src.noice_experiments.model import (
    FakeModel
)


class Ensemble:
    def __init__(self, grid, noise_cases, observations, path_to_forecasts, stations_to_out, error):
        '''
        Initialize an ensemble of FakeModels
        :param grid: CSVGridFile object with grid of model parameters
        :param noise_cases: List of noise case: 1 FakeModel instance per case
        :param observations: List of time series that correspond to observations
        :param path_to_forecasts: Directory where all noisy cases are located
        :param stations_to_out: List of station's indexes where to forecast
        '''

        self.grid = grid
        self.observations = observations
        self.forecasts_dir = path_to_forecasts
        self.stations_to_out = stations_to_out
        self.error = error
        self.models = self._initialized_models(noise_cases=noise_cases)

    def _initialized_models(self, noise_cases):
        models = []
        for noise in noise_cases:
            models.append(FakeModel(grid_file=self.grid, observations=self.observations,
                                    stations_to_out=self.stations_to_out, error=self.error,
                                    forecasts_path=os.path.join(self.forecasts_dir), noise_run=noise))

        return models

    def output(self, params):
        predictions = [model.output(params=params) for model in self.models]

        best_amount = 7
        predictions_best = sorted(predictions, key=lambda p: np.mean(p))[:best_amount]

        statistics_by_stations = {}
        for station_idx in range(len(self.stations_to_out)):
            predictions_for_station = column(predictions_best, station_idx)

            statistics_by_stations[station_idx] = {'std': np.std(predictions_for_station),
                                                   'mean': np.mean(predictions_for_station),
                                                   'max': np.max(predictions_for_station)}

        out = []
        for station in statistics_by_stations.keys():
            mean = statistics_by_stations[station]['mean']
            out.append(mean)

        return out

    def closest_params(self, params):
        drf = min(self.grid.drf_grid, key=lambda val: abs(val - params.drf))
        cfw = min(self.grid.cfw_grid, key=lambda val: abs(val - params.cfw))
        stpm = min(self.grid.stpm_grid, key=lambda val: abs(val - params.stpm))

        return drf, cfw, stpm


def column(matrix, idx):
    return [row[idx] for row in matrix]
