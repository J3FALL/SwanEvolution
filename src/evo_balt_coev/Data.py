import ast
import csv
import os

import pandas as pd

import Consts
import Log
import Supply


def clearData():
    dir = Consts.Models.SWAN.pathToExpNotesFolder
    for file in os.listdir(dir):
        try:
            os.remove(dir + file)
        except OSError:
            Log.write('Cant delete old data')


def newCSV(modelDate):
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + ".csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow([' ID '] + ['Pop'] + [' finErrorDist '] + [' params '] + [' errors '] + [' forecasts'])
    file.close()

    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations.csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow([' ID '] + ['Pop'] + [' finErrorDist '] + [' params '] + [' errors '] + [' forecasts'])
    file.close()

    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-bestinpopulations.csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow([' ID '] + ['Pop'] + [' finErrorDist '] + [' params '] + [' errors '] + [' forecasts'])
    file.close()

    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations-coev.csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow(
        [' Pop '] + [' coevError '] + [' Coeff_0 '] + [' Coeff_1 '] + [' ID_0 '] + ['Pop_0'] + [' finErrorDist_0 '] + [
            ' params_0 '] + [' errors_0 '] + [' ID_1 '] + ['Pop_1'] + [
            ' finErrorDist_1 '] + [' params_1 '] + [' errors_1 '] + [' forecasts_0'] + [' forecasts_1'])

    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations-coev-front.csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow(
        [' Pop '] + [' coevError '] + [' Coeff_0 '] + [' Coeff_1 '] + [' ID_0 '] + ['Pop_0'] + [' finErrorDist_0 '] + [
            ' params_0 '] + [' errors_0 '] + [' ID_1 '] + ['Pop_1'] + [
            ' finErrorDist_1 '] + [' params_1 '] + [' errors_1 '] + [' forecasts_0'] + [' forecasts_1'])

    file.close()


def writeIndiviual(modelDate, ind):
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + ".csv", 'ab')
    wr = csv.writer(file)
    wr.writerow([ind.ID, ind.pop, Supply.fullErrorFunction(ind), ind.params, ind.fullErrors,
                 ','.join([str(d) for d in ind.forecasts])])
    file.close()


def writeFront(modelDate, ind):
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-front.csv", 'ab')
    wr = csv.writer(file)
    if ind == 0:
        wr.writerow(['NEW GENERATION'])
    else:
        wr.writerow([ind.ID, ind.pop, Supply.fullErrorFunction(ind), ind.params, ind.fullErrors,
                     ','.join([str(d) for d in ind.forecasts])])
    file.close()
    return 0


def writePopulation(modelDate, individuals):
    for ind in individuals:
        ind.pop = Supply.ModelDate.numOfPopulation
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations.csv", 'ab')
    wr = csv.writer(file)
    map(lambda ind: wr.writerow([ind.ID, ind.pop, Supply.fullErrorFunction(ind), ind.params, ind.fullErrors,
                                 ','.join([str(d) for d in ind.forecasts])]), individuals)
    file.close()

    writeBestInPopulation(modelDate, individuals)

    return 0


def writeBestInPopulation(modelDate, individuals):
    def compare(item1, item2):
        if Supply.fullErrorFunction(item1) < Supply.fullErrorFunction(item2):
            return -1
        elif Supply.fullErrorFunction(item1) > Supply.fullErrorFunction(item2):
            return 1
        else:
            return 0

    for ind in individuals:
        ind.pop = Supply.ModelDate.numOfPopulation
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-bestinpopulations.csv", 'ab')
    wr = csv.writer(file)
    ind = sorted(individuals, cmp=compare)[0]

    wr.writerow([int(ind.ID), ind.pop, Supply.fullErrorFunction(ind), ind.params, ind.fullErrors,
                 ','.join([str(d) for d in ind.forecasts])])
    file.close()
    return 0


def readPopulations(popId, folderName, fileName):
    try:
        data = pd.read_csv(folderName + fileName + '.csv',
                           names=['ID', 'Pop', 'finErrorDist', 'params', 'errors', 'forecasts'])
        individuals = [Consts.individual(data.ID[i], data.Pop[i],
                                         ([ast.literal_eval(data.errors[i])[
                                               Consts.State.currentStationId]] if Consts.State.separateStationsMode else ast.literal_eval(
                                             data.errors[i])),
                                         ast.literal_eval(data.errors[i]), ast.literal_eval(data.params[i]),
                                         ast.literal_eval(data.forecasts[i])) for i in range(1, len(data.Pop)) if
                       data.Pop[i] == str(popId)]
        return individuals
    except IOError:
        Log.write('File {0}.csv does not exist.'.format(fileName))
        return []


def writeCoevolution(modelDate, coindividualds):
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations-coev.csv", 'ab')

    wr = csv.writer(file)
    # wr.writerow([' ID_0 '] + ['Pop_0'] + [' finErrorDist_0 '] + [' params_0 '] + [' errors_0 '] + [' ID_1 '] + ['Pop_1'] + [' finErrorDist_1 '] + [' params_1 '] + [' errors_1 '] )
    map(lambda coind: wr.writerow(
        [coind.Pop, coind.coevError, coind.couple[0].ensCoeff, coind.couple[1].ensCoeff, coind.couple[0].ID,
         coind.couple[0].pop, Supply.fullErrorFunction(coind.couple[0]), coind.couple[0].params,
         coind.couple[0].fullErrors, coind.couple[1].ID,
         coind.couple[1].pop, Supply.fullErrorFunction(coind.couple[1]), coind.couple[1].params,
         coind.couple[1].fullErrors, coind.couple[0].forecasts, coind.couple[1].forecasts]), coindividualds)
    file.close()
    return 0


def writeCoevolutionFront(modelDate, coindividualds):
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations-coev-front.csv", 'ab')

    wr = csv.writer(file)
    # wr.writerow([' ID_0 '] + ['Pop_0'] + [' finErrorDist_0 '] + [' params_0 '] + [' errors_0 '] + [' ID_1 '] + ['Pop_1'] + [' finErrorDist_1 '] + [' params_1 '] + [' errors_1 '] )

    map(lambda coind: wr.writerow(
        [coind.Pop, coind.coevError, coind.couple[0].ensCoeff, coind.couple[1].ensCoeff, coind.couple[0].ID,
         coind.couple[0].pop, Supply.fullErrorFunction(coind.couple[0]), coind.couple[0].params,
         coind.couple[0].fullErrors, coind.couple[1].ID,
         coind.couple[1].pop, Supply.fullErrorFunction(coind.couple[1]), coind.couple[1].params,
         coind.couple[1].fullErrors, coind.couple[0].forecasts, coind.couple[1].forecasts]), coindividualds)
    file.close()
    return 0
