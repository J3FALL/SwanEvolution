from ast import literal_eval
from collections import Counter

import numpy as np

from src.evo_balt.evo import PhysicsType
from src.evo_balt.evo import SWANParams


class FakeModel:
    '''
    Class that imitates SWAN-model behaviour:
        it encapsulates simulation results on a model params grid:
            [drag, physics, wcr, ws] = model_output, i.e. forecasts
    '''

    def __init__(self, config):
        '''
        Init parameters grid
        :param config: EvoConfig that contains parameters of evolution
        '''
        self.config = config
        self._init_grids()

    def _init_grids(self):
        self.drag_grid = self._grid(self.config.grid_by_name('drag'))
        self.wcr_grid = self._grid(self.config.grid_by_name('wcr'))
        self.ws_grid = self._grid(self.config.grid_by_name('ws'))

        self.grid = np.zeros((len(self.drag_grid), len(PhysicsType), len(self.wcr_grid), len(self.ws_grid)))
        # TODO: load grid values from smth

    def _grid(self, grid_params):
        grid = np.linspace(grid_params['min'], grid_params['max'],
                           num=int((grid_params['max'] - grid_params['min']) / grid_params['delta']))
        grid = np.round(grid, decimals=3)
        return grid

    def output(self, params):
        '''

        :param params: SWAN parameters
        :return: ForecastFile
        '''

        drag_idx = self.drag_grid.tolist().index(params.drag_func)
        physics_idx = (list(PhysicsType)).index(params.physics_type)
        wcr_idx = self.wcr_grid.tolist().index(params.wcr)
        ws_idx = self.ws_grid.tolist().index(params.ws)

        return self.grid[drag_idx, physics_idx, wcr_idx, ws_idx]


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
        with open(self.path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            grid_rows = [GridRow(row) for row in reader]

            drag_values = [row.model_params.drag_func for row in grid_rows]
            physics_types = [row.model_params.physics_type for row in grid_rows]
            wcr_values = [row.model_params.wcr for row in grid_rows]
            ws_values = [row.model_params.ws for row in grid_rows]
            print(self._grid_space(drag_values))
            print(self._grid_space(wcr_values))
            print(self._grid_space(ws_values))
            print(self._grid_space(physics_types))

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


grid = GridFile(path="../../samples/fixed.csv")
# fake = FakeModel(config=EvoConfig())
# print(fake.output(params=SWANIndivid(1.505, PhysicsType.GEN3, 1.505, 1.505)))
