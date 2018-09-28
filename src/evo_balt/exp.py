from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from evo import SPEA2
from model import GridFile, FakeModel
from src.evo_balt.files import ObservationFile

grid = GridFile(path="../../samples/grid_era_full.csv")
fake = FakeModel(grid_file=grid)

history = SPEA2(1000, 30, 10, 0.9).solution()
params = history.last().genotype
errors = fake.output(params)
forecasts = []
for row in grid.rows:
    if set(row.model_params.params_list()) == set(params.params_list()):
        forecasts = row.forecasts
        print("index : %d" % grid.rows.index(row))

waves_1 = ObservationFile(path="../../samples/obs/1a_waves.txt").time_series(from_date="20140814.120000",
                                                                             to_date="20140915.000000")
waves_2 = ObservationFile(path="../../samples/obs/2a_waves.txt").time_series(from_date="20140814.120000",
                                                                             to_date="20140915.000000")
waves_3 = ObservationFile(path="../../samples/obs/3a_waves.txt").time_series(from_date="20140814.120000",
                                                                             to_date="20140915.000000")

fig, axs = plt.subplots(2, 2)

time = np.linspace(1, 253, num=len(forecasts[0]))
axs[0, 0].plot(time, waves_1, label='Observations, Station 1')
axs[0, 0].plot(time, forecasts[0], label='Predicted, Station 1')
axs[0, 0].legend()
axs[0, 1].plot(time, waves_2, label='Observations, Station 2')
axs[0, 1].plot(time, forecasts[1], label='Predicted, Station 2')
axs[0, 1].legend()
axs[1, 0].plot(time, waves_3, label='Observations, Station 3')
axs[1, 0].plot(time, forecasts[2], label='Predicted, Station 3')
axs[1, 0].legend()

gens = [error.genotype_index for error in history.history]
error_vals = [error.error_value for error in history.history]

axs[1, 1].plot()
axs[1, 1].plot(gens, error_vals, label='Loss history', marker=".")
axs[1, 1].legend()

plt.show()


def error(row):
    return sqrt(pow(row.errors[0], 2) + pow(row.errors[1], 2) + pow(row.errors[2], 2))


print(min([error(row) for row in grid.rows]))
