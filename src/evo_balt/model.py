import os
import random
from ast import literal_eval
from collections import Counter
from enum import IntEnum
from math import sqrt

import numpy as np

OBSERVED_STATIONS = 3

from src.evo_balt.files import ObservationFile

wcr_range = [4.425e-10, 8.85e-10, 1.3275e-09, 4.425e-09, 8.85e-09, 1.3275e-08, 4.425e-08, 8.85e-08, 1.3275e-07,
             4.425e-07, 8.85e-07, 1.3275000000000002e-06, 4.425e-06, 8.85e-06, 1.3275e-05, 4.425e-05, 8.85e-05,
             0.00013275, 0.00044249999999999997, 0.0008849999999999999, 0.0013275000000000001, 0.004425, 0.00885,
             0.013275000000000002, 0.04425, 0.0885, 0.13275, 0.4425, 0.885, 1.3274999999999997]


class SWANParams:
    @staticmethod
    def new_instance():
        return SWANParams(drag_func=random.uniform(0, 5), physics_type=PhysicsType.GEN3,
                          wcr=random.choice(wcr_range), ws=0.00302)

    def __init__(self, drag_func, physics_type, wcr, ws):
        '''
        Represents parameters of SWAN model that will be evolve
        :param drag_func:
        :param physics_type: GEN1, GEN3 - enum ?
        :param wcr: WhiteCappingRate
        :param ws: Wave steepness
        '''

        self.drag_func = drag_func
        self.physics_type = physics_type
        self.wcr = wcr
        self.ws = ws

    def update(self, drag_func, physics_type, wcr, ws):
        self.drag_func = drag_func
        self.physics_type = physics_type
        self.wcr = wcr
        self.ws = ws

    def params_list(self):
        return [self.drag_func, self.physics_type, self.wcr, self.ws]


class PhysicsType(IntEnum):
    GEN1 = 0
    GEN3 = 1


class FakeModel:
    '''
    Class that imitates SWAN-model behaviour:
        it encapsulates simulation results on a model params grid:
            [drag, physics, wcr, ws] = model_output, i.e. forecasts
    '''

    def __init__(self, grid_file):
        '''
        Init parameters grid
        :param config: EvoConfig that contains parameters of evolution
        '''
        self.grid_file = grid_file
        self._init_grids()

        self.observations = self.observations_from_files()

    def _init_grids(self):
        self.grid = self._empty_grid()
        for row in self.grid_file.rows:
            drag_idx, physics_idx, wcr_idx, ws_idx = self.params_idxs(row.model_params)
            # self.grid[drag_idx, physics_idx, wcr_idx, ws_idx] = row.errors
            self.grid[drag_idx, physics_idx, wcr_idx, ws_idx] = \
                [FakeModel.Forecast(station_idx=idx, grid_row=row) for idx in range(OBSERVED_STATIONS)]

    def _empty_grid(self):
        return np.empty((len(self.grid_file.drag_grid), len(PhysicsType),
                         len(self.grid_file.wcr_grid),
                         len(self.grid_file.ws_grid)),
                        dtype=list)

    def observations_from_files(self):
        return [ObservationFile(path="../../samples/obs/1a_waves.txt").time_series(from_date="20140814.120000",
                                                                                   to_date="20140915.000000"),
                ObservationFile(path="../../samples/obs/2a_waves.txt").time_series(from_date="20140814.120000",
                                                                                   to_date="20140915.000000"),
                ObservationFile(path="../../samples/obs/3a_waves.txt").time_series(from_date="20140814.120000",
                                                                                   to_date="20140915.000000")
                ]

    def params_idxs(self, params):
        drag_idx = self.grid_file.drag_grid.index(params.drag_func)
        physics_idx = (list(PhysicsType)).index(params.physics_type)
        wcr_idx = self.grid_file.wcr_grid.index(params.wcr)
        ws_idx = self.grid_file.ws_grid.index(params.ws)

        return drag_idx, physics_idx, wcr_idx, ws_idx

    def closest_params(self, params):
        drag = min(self.grid_file.drag_grid, key=lambda val: abs(val - params.drag_func))
        physics = (list(PhysicsType)).index(params.physics_type)
        wcr = min(self.grid_file.wcr_grid, key=lambda val: abs(val - params.wcr))
        ws = min(self.grid_file.ws_grid, key=lambda val: abs(val - params.ws))

        return drag, physics, wcr, ws

    def output(self, params):
        '''

        :param params: SWAN parameters
        :return: List of forecasts for each station
        '''
        drag_idx, physics_idx, wcr_idx, ws_idx = self.params_idxs(params=params)
        return [self.error(forecast) for forecast in self.grid[drag_idx, physics_idx, wcr_idx, ws_idx]]

    class Forecast:
        def __init__(self, station_idx, grid_row):
            assert station_idx < len(grid_row.forecasts)

            self.station_idx = station_idx
            self.hsig_series = self._station_series(grid_row)

        def _station_series(self, grid_row):
            return grid_row.forecasts[self.station_idx]

    def error(self, forecast):
        '''
        Calculate RMSE of betwenn forecasts and observations for corresponding station
        :param forecast: Forecast object
        '''

        observation = self.observations[forecast.station_idx]

        result = 0.0
        for pred, obs in zip(forecast.hsig_series, observation):
            result += pow(pred - obs, 2)

        return sqrt(result)


class GridFile:
    '''
    Class that loads results of multiple SWAN simulations from CSV-file
    and construct grid of parameters
    '''

    def __init__(self, path):
        '''

        :param path: path to CSV-file
        '''
        self.path = path
        self._load()

    def _load(self):
        import csv
        with open(os.path.join(os.path.dirname(__file__), self.path), newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            grid_rows = [GridRow(row) for row in reader]
            self.rows = grid_rows

            drag_values = [row.model_params.drag_func for row in grid_rows]
            physics_types = [row.model_params.physics_type for row in grid_rows]
            wcr_values = [row.model_params.wcr for row in grid_rows]
            ws_values = [row.model_params.ws for row in grid_rows]

            self.drag_grid = self._grid_space(drag_values)
            self.wcr_grid = self._grid_space(wcr_values)
            self.ws_grid = self._grid_space(ws_values)
            self.physics_grid = self._grid_space(physics_types)

            print(self.drag_grid)
            print(self.wcr_grid)
            print(self.ws_grid)
            print(self.physics_grid)

    def _grid_space(self, param_values):
        '''
        Find parameter space of values counting all unique values
        '''

        cnt = Counter(param_values)
        return list(cnt.keys())


class GridRow:
    def __init__(self, row):
        self.id = row['ID']
        self.pop = row['Pop']
        self.error_distance = row['finErrorDist']
        self.model_params = self._swan_params(row['params'])
        self.errors = literal_eval(row['errors'])
        self.forecasts = literal_eval(row['forecasts'])

    def _swan_params(self, params_str):
        params_tuple = literal_eval(params_str)
        return SWANParams(drag_func=params_tuple[0], physics_type=params_tuple[1],
                          wcr=params_tuple[2], ws=params_tuple[3])
