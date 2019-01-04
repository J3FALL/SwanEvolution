from functools import partial
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from model import GridFile, FakeModel, SWANParams
from src.algorithm.spea2 import SPEA2
from src.evo_balt.files import ObservationFile
from src.evo_balt.swan_evo import calculate_objectives, crossover, mutation

grid = GridFile(path="../../samples/grid_era_full.csv")
fake = FakeModel(grid_file=grid)


def optimize():
    history = SPEA2(params=SPEA2.Params(max_gens=50, pop_size=20, archive_size=10, crossover_rate=0.5),
                    new_individ=SWANParams.new_instance,
                    objectives=partial(calculate_objectives, fake),
                    crossover=crossover,
                    mutation=mutation).solution()

    params = history.last().genotype

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

    return history


def error(row):
    return sqrt(pow(row.errors[0], 2) + pow(row.errors[1], 2) + pow(row.errors[2], 2))


def find_min_error(model, grid):
    optimal = pow(10, 9)
    opt_row = -1
    for row in grid.rows:
        stations = [FakeModel.Forecast(station_idx=0, grid_row=row),
                    FakeModel.Forecast(station_idx=1, grid_row=row),
                    FakeModel.Forecast(station_idx=2, grid_row=row)]
        err = \
            sqrt(pow(model.error(stations[0]), 2) + pow(model.error(stations[1]), 2) + pow(model.error(stations[2]), 2))
        if err < optimal:
            optimal = err
            opt_row = row
    return optimal, opt_row


def plot_rmse_surface(model):
    history = optimize()
    print([point.error_value for point in history.history])

    drags = []
    wcrs = []
    errors = []
    for point in history.history:
        drags.append(point.genotype.drag_func)
        wcrs.append(point.genotype.wcr)
        errors.append(point.error_value)

    x = np.zeros(len(model.grid_file.drag_grid))
    y = np.zeros(len(model.grid_file.wcr_grid))
    z = np.zeros((len(y), len(x)), dtype=np.float32)
    points = []
    for drag_idx in range(len(model.grid_file.drag_grid)):
        for wcr_idx in range(len(model.grid_file.wcr_grid)):
            forecasts = model.grid[drag_idx, 1, wcr_idx, 0]
            rmse = \
                sqrt(pow(model.error(forecasts[0]), 2) + pow(model.error(forecasts[1]), 2) +
                     pow(model.error(forecasts[2]), 2))

            points.append([model.grid_file.drag_grid[drag_idx], model.grid_file.wcr_grid[wcr_idx], rmse])
            x[drag_idx] = model.grid_file.drag_grid[drag_idx]
            y[wcr_idx] = model.grid_file.wcr_grid[wcr_idx]
            z[wcr_idx, drag_idx] = rmse

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()

    ax = Axes3D(fig)

    cset = ax.plot_surface(X, np.log10(Y), z, rstride=1, cstride=1,
                           cmap='viridis', edgecolor='none')

    ax.plot(drags, np.log10(wcrs), errors, marker='o', markersize=10)
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_xlabel('drag')
    ax.set_ylabel('wcr')
    ax.set_zlabel('rmse')
    plt.show()


def plot_sensitivity():
    drag_values = grid.drag_grid
    station_1 = []
    station_2 = []
    station_3 = []
    rmse_full = []
    for drag_idx in range(len(drag_values)):
        forecasts = fake.grid[drag_idx, 1, 0, 0]

        station_1.append(fake.error(forecasts[0]))
        station_2.append(fake.error(forecasts[1]))
        station_3.append(fake.error(forecasts[2]))

        rmse_full.append(sqrt(pow(station_1[-1], 2) + pow(station_2[-1], 2) + pow(station_3[-1], 2)))

    plt.figure()
    plt.plot(drag_values, station_1, label='Station 1, RMSE')
    plt.plot(drag_values, station_2, label='Station 2, RMSE')
    plt.plot(drag_values, station_3, label='Station 3, RMSE')
    plt.plot(drag_values, rmse_full, label='Full RMSE')
    plt.legend()
    plt.show()


# plot_sensitivity()
plot_rmse_surface(fake)
