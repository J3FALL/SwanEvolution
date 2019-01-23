import os

from src.noice_experiments.errors import error_rmse_peak
from src.noice_experiments.model import (
    FakeModel,
    CSVGridFile
)
from src.swan.files import real_obs_from_files


class Ensemble:
    def __init__(self, grid, noise_cases, observations, path_to_forecasts, stations_to_out):
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
        self.models = self._initialized_models(noise_cases=noise_cases)

    def _initialized_models(self, noise_cases):
        models = []
        for noise in noise_cases:
            models.append(FakeModel(grid_file=self.grid, observations=self.observations,
                                    stations_to_out=self.stations_to_out, error=error_rmse_peak,
                                    forecasts_path=os.path.join(self.forecasts_dir, str(noise)), noise_run=noise))

        return models


grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
obs = [obs.time_series(from_date="20140814.120000", to_date="20140915.000000") for obs in real_obs_from_files()]

ens = Ensemble(grid=grid, noise_cases=[1, 5, 25, 26], observations=obs,
               path_to_forecasts='../../../wind-noice-runs/results_fixed',
               stations_to_out=[1, 2, 3])
