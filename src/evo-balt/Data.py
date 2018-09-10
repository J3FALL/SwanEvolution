import csv, Supply, os, Log, Consts
import pandas as pd
import ast

def clearData():
    dir = Consts.Models.SWAN.pathToExpNotesFolder
    for file in os.listdir(dir):
        try:
            os.remove(dir + file)
        except OSError:
            Log.write('Cant delete old data')

def newCSV(modelDate):
    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + ".csv", 'wb')
    wr = csv.writer(file, dialect='excel')#, quoting=csv.QUOTE_ALL
    wr.writerow([' ID '] + ['Pop'] + [' finErrorDist '] +  [' params '] + [' errors ']+ [' forecasts'])
    file.close()

    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-populations.csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow([' ID '] + ['Pop'] + [' finErrorDist '] + [' params '] + [' errors '] + [' forecasts'])
    file.close()

    file = open(Consts.Models.SWAN.pathToExpNotesFolder + Consts.State.expId + "-bestinpopulations.csv", 'wb')
    wr = csv.writer(file, dialect='excel')  # , quoting=csv.QUOTE_ALL
    wr.writerow([' ID '] + ['Pop'] + [' finErrorDist '] + [' params '] + [' errors '] + [' forecasts'])
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

    wr.writerow([int(ind.ID), ind.pop, Supply.fullErrorFunction(ind),  ind.params, ind.fullErrors, ','.join([str(d) for d in ind.forecasts])])
    file.close()
    return 0


def readPopulations(popId, folderName, fileName):
    try:
        data = pd.read_csv(folderName+fileName+'.csv', names=['ID', 'Pop', 'finErrorDist', 'params', 'errors', 'forecasts'])
        individuals = [Consts.individual(data.ID[i], data.Pop[i],
                                         ([ast.literal_eval(data.errors[i])[Consts.State.currentStationId]] if Consts.State.separateStationsMode else ast.literal_eval(data.errors[i])),
                                         ast.literal_eval(data.errors[i]), ast.literal_eval(data.params[i]),
                                         ast.literal_eval(data.forecasts[i])) for i in range(1,len(data.Pop)) if data.Pop[i]==str(popId)]
        return individuals
    except IOError:
        Log.write('File {0}.csv does not exist.'.format(fileName))
        return []