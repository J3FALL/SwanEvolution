import random
import subprocess
from datetime import datetime

import numpy as np

import Consts
import Data
import Log
import SwanFunctions
from src.evo_old.files import ForecastFile
from src.evo_old.files import ObservationFile


class ModelDate:
    Year = 2014
    Month = 8
    Days = 14
    Hours = 12
    numOfIndividuals = 0
    numOfPopulation = 1


def getDate(y, m, d, h, delimiter):
    s = '{0:02d}{1:02d}{2:02d}{3:02d}'.format(y, m, d, h)  # str(y) + str(m) + str(d) + str(h) #'2013102000'
    if delimiter == 0:
        mytime = datetime.strptime(s, "%Y%m%d%H")
        return mytime.strftime("%Y%m%d%H")
    else:
        mytime = datetime.strptime(s, "%Y%m%d%H")  # 2013-10-20T12:00:00
        return mytime.strftime("%Y-%m-%dT%H") + ":00:00"


def getForecast(station, colNum):
    if Consts.Debug.debugMode:
        return [(random.random() * 5)] * 1000

    pathEst = Consts.Models.SWAN.pathToResults + station

    forecast = ForecastFile(path=pathEst)
    content = forecast.time_series()

    return [float(line.split()[colNum]) for line in content]


def getObservation(station, colNum):  # returnable value's length is equal to meteoForecastTime + 1 !!!

    pathObs = Consts.Models.Observations.pathToFolder + station

    obs_file = ObservationFile(path=pathObs)
    content = obs_file.time_series(
        from_date=Consts.Models.Observations.timePeriodsStartTimes[Consts.State.currentPeriodId],
        to_date=Consts.Models.Observations.timePeriodEndTimes[Consts.State.currentPeriodId])

    return [float(line.split()[colNum]) for line in content]


def parseDate(modelDate):  # 2013-10-20T12-00-00
    if ('-' in modelDate):
        date = modelDate.split('-')[:2]
        date.extend([modelDate.split('-')[2].split('T')[0], modelDate.split('-')[2].split('T')[1]])
        return map(lambda d: int(d), date)
    if ('.' in modelDate):
        date = modelDate
        return [int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[9:11])]


def runBatFile():
    p = subprocess.Popen('D:\\EvoBalt_v3.0-single\\runBatFile.exe')
    p.wait()


def getMultidimDistance(dims1, dims2):
    result = 0
    for i in range(0, len(dims1)):
        diffItem = (float(dims1[i]) - float(dims2[i])) ** 2
        result += diffItem
    return math.sqrt(result)


def errorFunction(item):
    return getMultidimDistance(item.errors, [0] * len(item.errors))


def fullErrorFunction(item):
    return getMultidimDistance(item.fullErrors, [0] * len(item.fullErrors))


def runModel(params, modelDate):
    Log.write('Run model with params {0} and date {1}'.format(params, modelDate))

    # start-end-time
    SwanFunctions.writeSwanConfig(Consts.Models.SWAN.pathToConfig(), "COMPUTE",
                                  [2, 5],
                                  [Consts.Models.Observations.timePeriodsStartTimes[Consts.State.currentPeriodId],
                                   Consts.Models.Observations.timePeriodEndTimes[Consts.State.currentPeriodId]])

    SwanFunctions.writeSwanConfig(Consts.Models.SWAN.pathToConfig(), "OUTput",
                                  [1],
                                  Consts.Models.Observations.timePeriodsStartTimes[Consts.State.currentPeriodId])
    # drag
    SwanFunctions.writeSwanConfig(Consts.Models.SWAN.pathToConfig(), Consts.Models.SWAN.Parameters.WindCoeff.name,
                                  np.asarray([Consts.Models.SWAN.Parameters.WindCoeff.valueColumnId]),
                                  str(params[Consts.Models.SWAN.Parameters.WindCoeff.indInParamsArray]))

    # GEN
    physicsTypeName = Consts.Models.SWAN.Parameters.PhysicsType.typesNames[
        (params[Consts.Models.SWAN.Parameters.PhysicsType.indInParamsArray])]
    SwanFunctions.writeSwanConfig(Consts.Models.SWAN.pathToConfig(), Consts.Models.SWAN.Parameters.PhysicsType.name,
                                  np.asarray([Consts.Models.SWAN.Parameters.PhysicsType.valueColumnId]),
                                  physicsTypeName)
    # wcr
    SwanFunctions.writeSwanConfig(Consts.Models.SWAN.pathToConfig(),
                                  Consts.Models.SWAN.Parameters.WhiteCappingRate.name,
                                  np.asarray([Consts.Models.SWAN.Parameters.WhiteCappingRate.valueColumnId]),
                                  Consts.Models.SWAN.Parameters.WhiteCappingRate.name + str(
                                      params[Consts.Models.SWAN.Parameters.WhiteCappingRate.indInParamsArray]))
    # ws
    SwanFunctions.writeSwanConfig(Consts.Models.SWAN.pathToConfig(), Consts.Models.SWAN.Parameters.WaveSteepness.name,
                                  np.asarray([Consts.Models.SWAN.Parameters.WaveSteepness.valueColumnId]),
                                  Consts.Models.SWAN.Parameters.WaveSteepness.name + str(
                                      Consts.Models.SWAN.Parameters.WaveSteepness.defaultValue))  # str(params[Consts.Models.SWAN.Parameters.WaveSteepness.indInParamsArray]))

    if not Consts.Debug.debugMode:
        SwanFunctions.runSwanBatFile()  # run simulation
    ModelDate.numOfIndividuals += 1  # ID

    observationsAtStations = [0] * Consts.Models.Observations.Stations.FullCount
    forecastAtStations = [0] * Consts.Models.Observations.Stations.FullCount
    errorAtStations = [0] * Consts.Models.Observations.Stations.FullCount

    for stationId in range(0, Consts.Models.Observations.Stations.FullCount):
        observationsAtStations[stationId], forecastAtStations[stationId], errorAtStations[stationId] = calculateErrors(
            Consts.Models.SWAN.Stations.FullNames[stationId], Consts.Models.SWAN.OutputColumns.Hsig,
            Consts.Models.Observations.Stations.FullNames[stationId],
            Consts.Models.Observations.OutputColumns.Hsig)

    selectedErrors = errorAtStations
    if (Consts.State.separateStationsMode):
        selectedErrors = [errorAtStations[Consts.State.currentStationId]]

    individuals = [
        Consts.individual(ModelDate.numOfIndividuals, ModelDate.numOfPopulation, selectedErrors, errorAtStations,
                          params, forecastAtStations)]
    Data.writeIndiviual(modelDate, individuals[-1])  # write into csv data about individual
    # States.copy(individuals[-1].ID, '_zi-all.txt')
    # States.copy(individuals[-1].ID, '_vel-all.txt')
    return individuals[-1]  # copy.deepcopy(individuals[-1])


def calculateErrors(station, point, station_obs, point_obs):
    est = getForecast(station, point)
    obs = getObservation(station_obs, point_obs)
    time_range = len(obs)

    return [obs, est, np.sqrt(sum(map(lambda y, x: (x - y) ** 2, obs[:time_range], est[:time_range])) / time_range)]
