import copy
import datetime
import math

import numpy as np

import Consts
import Data
import Log
import Supply
import SwanFunctions


class Coindividual:
    def __init__(self, Pop, couple, errors, coevError):
        self.Pop = Pop
        self.couple = couple
        self.errors = errors
        self.coevError = coevError


def getCoevError(couples, station, point, station_obs, point_obs):
    Log.write('Assess individuals')
    # w0, w1 = countCoevParameters()
    print("Current time when coefficients are computed :  {0}".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # print "Params w0 = {0:0.4f}, w1 = {1:0.4f}".format(w0, w1)
    return getErrors(couples, station, point, station_obs, point_obs)


def getBestIndividuals(individuals, anotherIndividuals, modelDate):
    Log.write('Get best individuals')
    couples = mix(copy.deepcopy(individuals), copy.deepcopy(anotherIndividuals))
    errorAtStations = np.zeros((len(couples), Consts.Models.Observations.Stations.FullCount))
    # np.asarray([[0] * Consts.Models.Observations.Stations.FullCount]*len(couples))
    # for coupleId in range(0,len(couples)):
    for stationId in range(0, Consts.Models.Observations.Stations.FullCount):
        res = getCoevError(couples, Consts.Models.SWAN.Stations.FullNames[stationId],
                           Consts.Models.SWAN.OutputColumns.Hsig,
                           Consts.Models.Observations.Stations.FullNames[stationId],
                           Consts.Models.Observations.OutputColumns.Hsig)
        for coupleId in range(0, len(couples)):
            errorAtStations[coupleId, stationId] = res[coupleId]

    front, MSError = selection(modelDate,
                               map(lambda c, errors: Coindividual(Supply.ModelDate.numOfPopulation + 1, c, errors, -1),
                                   couples, errorAtStations))
    return front, MSError


def calculateErrors(station, point, station_obs, point_obs):
    est = Supply.getForecast(station, point)
    obs = Supply.getObservation(station_obs, point_obs)
    time_range = len(obs)

    # return [obs , est, sum(map(lambda y, x: (x-y)**2, obs, est)) / len(est)]
    return [obs, est, np.sqrt(sum(map(lambda y, x: (x - y) ** 2, obs[:time_range], est[:time_range])) / time_range)]


def getMultidimDistance(dims1, dims2):
    result = 0
    for i in range(0, len(dims1)):
        diffItem = (float(dims1[i]) - float(dims2[i])) ** 2
        result += diffItem
    return math.sqrt(result)


def getRawFitness(coindividuals, front, index):
    Log.write('Raw fitness coevolution')
    if index > len(front) - 1 or index < 0:
        print
        'Index Exception, index = {0}, len of individuals = {1}'.format(index, len(front))
        return 1
    return 1 + len(
        filter(lambda ind: front[index].errors[0] <= ind.errors[0] and front[index].errors[1] <= ind.errors[1],
               copy.deepcopy(coindividuals))) / len(coindividuals)


def getErrors(couples, station, point, station_obs, point_obs):
    obs = Supply.getObservation(station_obs, point_obs)
    time_range = len(obs)
    stationIndex = Consts.Models.SWAN.Stations.FullNames.index(station)
    sumVal = [np.sqrt(sum(
        map(lambda obs, est1, est2: (obs - (est1 * couple[0].ensCoeff + est2 * couple[1].ensCoeff)) ** 2,
            obs[:time_range], couple[0].forecasts[stationIndex][:time_range],
            couple[1].forecasts[stationIndex][:time_range])) / len(couples[0][0].forecasts[stationIndex][:time_range]))
              for couple in couples]
    return sumVal


def countCoevParameters():
    Log.write('Count parameters')

    # return [0.504, 1.075]
    return [0.508, 0.552]


def mix(individuals, anotherIndividuals):
    Log.write('Mix individuals')
    couplesOfIndividuals = []
    for i in individuals:
        for a in anotherIndividuals:
            couplesOfIndividuals.append([i, a])
    return couplesOfIndividuals


def selection(modelDate, coindividuals):
    Log.write('Selection coevolution')

    for i in range(0, len(coindividuals)):
        coindividuals[i].coevError = getMultidimDistance(coindividuals[i].errors, [0] * len(coindividuals[i].errors))

    Data.writeCoevolution(modelDate, coindividuals)

    coindivSorted = copy.deepcopy(coindividuals)
    # coindivSorted.extend(readArchive(modelDate, coindividuals))
    for i in range(0, Consts.Models.Observations.Stations.Count):
        coindivSorted = sorted(coindivSorted, key=lambda ind: ind.errors[i], reverse=1)

    coindivSorted = sorted(coindivSorted, key=lambda ind: ind.errors[0])

    front = [coindivSorted[0]]

    for i in range(1, len(coindivSorted)):
        for j in range(0, Consts.Models.Observations.Stations.Count):
            if (coindivSorted[i].errors[j] <= front[-1].errors[j]):
                front.append(coindivSorted[i])

    front = truncateFront(front, coindividuals)

    for i in range(0, len(front)):
        front[i].coevError = getMultidimDistance(front[i].errors, [0] * len(front[i].errors))

    meanError = min([(coind.coevError) for coind in front])

    Data.writeCoevolutionFront(modelDate, front)

    return front, meanError


def truncateFront(front, coindividuals):  # returns front which size is minimum from 1 to 4 and maximum sqrt(len(front))
    Log.write('Truncate front coevolution')
    kNearest = int(math.sqrt(len(front)))
    i = 1
    while kNearest + 2 < len(front):  # +2 as it is a distance between any two objects
        distances = [getMultidimDistance(front[i].errors, front[i - 1].errors) for i in range(1, len(front))]
        i = i + 1
        index = distances.index(min(distances))
        # print 'Len(dist) = {0}, len(coinds) = {1}, index = {2}, len(front) = {3} \n'.format(len(distances), len(coindividuals), index, len(front))
        if index - 1 < 0:
            del front[index + 1]
            continue
        if index + 1 == len(distances):
            del front[index]
            continue
        if getRawFitness(coindividuals, front, index) * distances[index - 1] > getRawFitness(coindividuals, front,
                                                                                             index + 1) * distances[
            index + 1]:
            del front[index + 1]
        else:
            del front[index]
    return front


def preprocess(coindividuals):
    # Log.write('Preprocess')
    modelDate = (SwanFunctions.readSwanConfig(Consts.Models.SWAN.pathToConfig(), 'COMPUTE',
                                              [2]))  # discover model date written in config.ini

    Supply.ModelDate.Year, Supply.ModelDate.Month, Supply.ModelDate.Days, Supply.ModelDate.Hours = Supply.parseDate(
        modelDate)  # get date of modelling
    Data.newCSV(modelDate)  # create a csv-file to write the entire data about each individual

    individuals1 = []  # clear list for future work
    individuals2 = []
    Consts.State.forcingId = 0
    for i in range(0, 7):
        individuals1.append(Supply.runModel([Consts.Models.SWAN.Parameters.WindCoeff.first[i], 1,
                                             Consts.Models.SWAN.Parameters.WhiteCappingRate.first[i],
                                             Consts.Models.SWAN.Parameters.WaveSteepness.defaultValue,
                                             Consts.Models.SWAN.Parameters.ensCoeff.first[i]], modelDate))
        individuals1[i].ensCoeff = Consts.Models.SWAN.Parameters.ensCoeff.first[i]
    Consts.State.forcingId = 1
    for i in range(0, 7):
        individuals2.append(Supply.runModel([Consts.Models.SWAN.Parameters.WindCoeff.first2[i], 1,
                                             Consts.Models.SWAN.Parameters.WhiteCappingRate.first2[i],
                                             Consts.Models.SWAN.Parameters.WaveSteepness.defaultValue,
                                             Consts.Models.SWAN.Parameters.ensCoeff.first[i]], modelDate))
        individuals2[i].ensCoeff = Consts.Models.SWAN.Parameters.ensCoeff.first2[i]
    Consts.State.forcingId = 0

    coindividuals, newError = getBestIndividuals(individuals1, individuals2, modelDate)

    return (modelDate, individuals1, individuals2, newError)
