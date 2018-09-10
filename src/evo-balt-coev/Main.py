import CoevolutionData
import Consts
import Data
import Log
import Supply
import copy
import math
import random as rnd
import time


def coevolution(modelDate, bestIndividuals1, bestIndividuals2):
    # savedForcingId = 0
    # Consts.State.forcingId = 1
    # bestIndividualsEra = [Supply.runModel(params, modelDate) for params in paramsList]
    # Consts.State.forcingId = savedForcingId

    coindividuals, newError = CoevolutionData.getBestIndividuals(bestIndividuals1, bestIndividuals2, modelDate)
    return newError, coindividuals


def crossover(individuals,
              convergenceExtent):  # neighborhood crossover, mating with k-Nearest solutions
    Log.write('Crossover')
    sign = lambda x: -1 if x < 0.5 else 1

    paramsList = []

    # dragFuncList = []
    # physicsTypeList = []

    realI = 0
    for i in range(Consts.Evolution.populationMaxSize):
        if (realI >= len(individuals)):
            realI = 0
        paramsList.append((copy.copy(individuals[realI].dragFunc), copy.copy(individuals[realI].physicsType),
                           copy.copy(individuals[realI].wcr), copy.copy(individuals[realI].ws),
                           copy.copy(individuals[realI].ensCoeff)))
        realI += 1

    newParamsList = []
    for param in paramsList:
        neighbors = getNeighbors(individuals, paramsList.index(param) % len(
            individuals))  # get neighbors that are alike to present solution in terms of errors

        drf = param[Consts.Models.SWAN.Parameters.WindCoeff.indInParamsArray]

        drf = math.fabs((drf + neighbors[rnd.randint(0, len(neighbors) - 1)].dragFunc + sign(
            rnd.random()) * rnd.random() * 0.55) / 2)

        ensCoeff = param[Consts.Models.SWAN.Parameters.ensCoeff.indInParamsArray]
        # ensCoeff = math.fabs((ensCoeff + neighbors[rnd.randint(0, len(neighbors) - 1)].ensCoeff + sign(
        #    rnd.random()) * rnd.random() * 0.25) / 4)
        ensCoeff = math.fabs(ensCoeff)

        physicsType = neighbors[rnd.randint(0, len(neighbors) - 1)].physicsType

        wcr = param[Consts.Models.SWAN.Parameters.WhiteCappingRate.indInParamsArray]
        ws = param[Consts.Models.SWAN.Parameters.WaveSteepness.indInParamsArray]

        if (physicsType == Consts.Models.SWAN.Parameters.PhysicsType.typesNames.index("GEN3")):  # only for GEN3

            wcr = math.fabs((wcr + neighbors[rnd.randint(0, len(neighbors) - 1)].wcr + sign(
                rnd.random()) * rnd.random() * 0.00001) / 2)
        # ws = math.fabs((ws + neighbors[rnd.randint(0, len(neighbors) - 1)].ws + sign(
        #    rnd.random()) * rnd.random() * 0.0001) / 2)

        newParamsList.append((drf, physicsType, wcr, ws, ensCoeff))
    return newParamsList


def mutation(individuals, convergenceExtent,
             forced=False):  # write variable influenced by convergence extent, the higher is extent the the higher mutation's coefficient is
    Log.write('Mutation')
    sign = lambda x: -1 if x < 0.5 else 1
    paramsList = []

    realI = 0
    for i in range(Consts.Evolution.populationMaxSize):
        if (realI >= len(individuals)):
            realI = 0

        paramsList.append((copy.copy(individuals[realI].dragFunc), copy.copy(individuals[realI].physicsType),
                           copy.copy(individuals[realI].wcr), copy.copy(individuals[realI].ws),
                           copy.copy(individuals[realI].ensCoeff)))
        realI += 1

    newParamsList = []
    add = 1
    if (forced): add = 4
    for param in paramsList:
        drf = param[Consts.Models.SWAN.Parameters.WindCoeff.indInParamsArray]
        drf += sign(rnd.random()) * 0.55 * rnd.random() * paramsList[rnd.randint(0, len(paramsList) - 1)][
            Consts.Models.SWAN.Parameters.WindCoeff.indInParamsArray] * add
        drf = math.fabs(drf)

        ensCoeff = param[Consts.Models.SWAN.Parameters.ensCoeff.indInParamsArray]
        # ensCoeff += sign(rnd.random()) * 0.15 * rnd.random() * paramsList[rnd.randint(0, len(paramsList) - 1)][
        #    Consts.Models.SWAN.Parameters.ensCoeff.indInParamsArray] * add
        ensCoeff = math.fabs(ensCoeff)

        physicsTypeMutator = rnd.random() * 4
        # if (physicsTypeMutator>3): physicsType = param[Consts.Models.SWAN.Parameters.PhysicsType.indInParamsArray] #stay the same
        # elif (physicsTypeMutator < 1): physicsType = Consts.Models.SWAN.Parameters.PhysicsType.typesNames.index("GEN1")
        # elif (physicsTypeMutator <= 3): physicsType = Consts.Models.SWAN.Parameters.PhysicsType.typesNames.index("GEN3")

        physicsType = Consts.Models.SWAN.Parameters.PhysicsType.typesNames.index("GEN3")

        wcr = param[Consts.Models.SWAN.Parameters.WhiteCappingRate.indInParamsArray]
        ws = param[Consts.Models.SWAN.Parameters.WaveSteepness.indInParamsArray]

        if (physicsType == Consts.Models.SWAN.Parameters.PhysicsType.typesNames.index("GEN3")):  # only for GEN3

            wcr += sign(rnd.random()) * 0.75 * rnd.random() * paramsList[rnd.randint(0, len(paramsList) - 1)][
                Consts.Models.SWAN.Parameters.WhiteCappingRate.indInParamsArray] * add
            wcr = math.fabs(wcr)

            # ws += sign(rnd.random()) * 0.25 * rnd.random() * paramsList[rnd.randint(0, len(individuals) - 1)][Consts.Models.SWAN.Parameters.WaveSteepness.indInParamsArray] * add
            # ws = math.fabs(ws)
        newParamsList.append((drf, physicsType, wcr, ws, ensCoeff))
    return newParamsList


def evolution(individuals1, individuals2, modelDate, convErrorOld, isConvergence=False, isCrossover=True,
              convergenceExtent=1, oldCoevError=10000):  # 10000 is just a number that can't exist):
    Log.write('Evolution')
    convergenceExtent = Consts.Evolution.convergenceExtent
    while not (isConvergence):
        Log.write('_____________NEW POPULATION_____________number {0}'.format(Supply.ModelDate.numOfPopulation + 1))
        paramsList1 = operation(individuals1, isCrossover, convergenceExtent)
        paramsList2 = operation(individuals2, isCrossover, convergenceExtent)
        if (len(paramsList1) > 0):
            coindividuals = []

        Consts.State.forcingId = 0
        individuals1 = ([Supply.runModel(params, modelDate) for params in paramsList1])
        Consts.State.forcingId = 1
        individuals2 = ([Supply.runModel(params, modelDate) for params in paramsList2])
        Consts.State.forcingId = 0

        if (not Consts.Debug.debugMode): time.sleep(3.0)

        coevError, newCoindividuals = coevolution(modelDate, individuals1, individuals2)

        individuals1 = [ind for ind in individuals1 if
                        ind.ID in [coind.couple[0].ID for coind in [coind for coind in newCoindividuals]]]
        individuals2 = [ind for ind in individuals2 if
                        ind.ID in [coind.couple[1].ID for coind in [coind for coind in newCoindividuals]]]
        convErrorNew = coevError  # min([Supply.getMultidimDistance(ind.errors, [0] * len(ind.errors)) for ind in individuals])

        # map(lambda ind: Data.writeFront(modelDate, ind), individuals)
        # min([getDistance(ind.eSt2, 0, ind.eSt1, 0) for ind in individuals])

        # map(lambda ind: Data.writeFront(modelDate, ind), individuals)

        # if convErrorNew - convErrorOld < -Consts.Evolution.errorBorderForExtent:  # branch condition to check the extent of convergence
        #    convErrorOld = convErrorNew
        #    convergenceExtent = 1
        # else:
        #    convergenceExtent += 1
        Log.write('Convergence criteria = {0}'.format(str(convErrorNew - convErrorOld)))
        if ((abs(
                convErrorNew / convErrorOld) > 1.5 and Supply.ModelDate.numOfPopulation >= Consts.Evolution.minPopulationsNumber) or Supply.ModelDate.numOfPopulation >= Consts.Evolution.maxPopulationsNumber):
            isConvergence = True

        convErrorOld = convErrorNew

        # Log.write('Convergence extent = {0}; Convergence error is {1}'.format(convergenceExtent, convErrorNew))

        # Log.write('Population size is {0}'.format(len(individuals)))

        # Log.write('Params funcs is {0}'.format(paramsList))
        # if convergenceExtent >= Consts.Evolution.criticalConvergenceExtent:
        #    isConvergence = True  # break while-loop

        isCrossover = not isCrossover

        Consts.Evolution.populationMaxSize = Consts.Evolution.populationMaxSize - 2
        if (Consts.Evolution.populationMaxSize < 4): Consts.Evolution.populationMaxSize = 4

        Supply.ModelDate.numOfPopulation += 1
        Log.write('Num of Pop = {0} ;  Convergence ?= {1}'.format(Supply.ModelDate.numOfPopulation, isConvergence))
    return newCoindividuals


def getNeighbors(individuals, index):
    Log.write('Get neighbors')
    kNearest = int(math.sqrt(len(individuals)))
    filteredIndividuals = filter(lambda ind: ind.ID != individuals[index].ID, copy.deepcopy(individuals))
    if len(
            filteredIndividuals) <= kNearest:  # i.e. if len(individuals) == 1 {len(filtered) = 0 < sqrt(1)}, hence there is no neighbors
        return individuals
    return sorted(filteredIndividuals,
                  key=lambda ind: Supply.getMultidimDistance(ind.errors, [0] * len(ind.errors)))[
           :kNearest]  # I also can make it in a probabilistic way: the closer solutinos are to initial point,
    #  the higher chance of mateing they have.


def checkDomanates(dominatorErrors, looserErrors):
    for i in range(0, len(dominatorErrors)):
        if (dominatorErrors[i] > looserErrors[i]):
            return False
    return True


def getDominatingSolutions(individuals, front,
                           index):  # return number of solutions present solution dominates .It can take values from 1 to 2.
    Log.write('Raw fitness')
    if index > len(front) - 1 or index < 0:
        print
        'Index Exception, index = {0}, len of individuals = {1}'.format(index, len(front))
        return 1
    return 1 + len(filter(lambda ind: checkDomanates(front[index].errors, ind.errors),
                          copy.deepcopy(individuals))) / len(individuals)  # adding 1 to avoid zero value


def operation(individuals, isCrossover, convergenceExtent):  # returns list of new drag functions
    if isCrossover:
        return crossover(individuals, convergenceExtent)
    else:
        return mutation(individuals, convergenceExtent)


'''
def preprocess(individuals):
    Log.write('Preprocess')
    modelDate = (SwanFunctions.readSwanConfig(Consts.Models.SWAN.pathToConfig(), 'COMPUTE',
                                              [2]))  # discover model date written in config.ini

    Supply.ModelDate.Year, Supply.ModelDate.Month, Supply.ModelDate.Days, Supply.ModelDate.Hours = Supply.parseDate(modelDate)  # get date of modelling
    Data.newCSV(modelDate)  # create a csv-file to write the entire data about each individual

    #bestIndividuals = copy.deepcopy(individuals) # - DA version

    del individuals[:]  # clear list for future work

    #individuals.append(default(modelDate))

   # individuals.append(runModel(modelDate))
    # run default as it has to be according to the strategy, and because it generate ATM and UV files
    #bestIndividuals = [individual(0, 0, 0, 0, 0, drf, 0, 0, 0, 0) for drf in
    #                   Supply.read_old_date_last_front(Supply.ModelDate.Year, Supply.ModelDate.Month,
    #                                                   Supply.ModelDate.Days,
     #                                                  Supply.ModelDate.Hours)]  # - FORECASAT version


    if (Consts.State.readInitialFromFile):
        individuals = Data.readPopulations(1,Consts.State.initialFolder,Consts.State.expId)
    else:
        for i in range(0,7):
            individuals.append(Supply.runModel([Consts.Models.SWAN.Parameters.WindCoeff.first[i],1,Consts.Models.SWAN.Parameters.WhiteCappingRate.first[i],Consts.Models.SWAN.Parameters.WaveSteepness.defaultValue],modelDate))



    Data.writePopulation(modelDate,individuals)
    #individuals  = (initializing(modelDate, [], individuals))

    return [copy.deepcopy(individuals), modelDate,
            min([Supply.getMultidimDistance(ind.errors, [0] * len(ind.errors)) for ind in individuals])]

'''


def selection(individuals, modelDate):
    Log.write('Selection')
    indivSorted = copy.deepcopy(individuals)

    for i in range(0, Consts.Models.Observations.Stations.Count):
        indivSorted = sorted(indivSorted, key=lambda ind: ind.errors[i], reverse=1)

    indivSorted = sorted(indivSorted, key=lambda ind: ind.errors[0])
    front = [indivSorted[0]]

    for i in range(1, len(indivSorted)):
        for j in range(0, Consts.Models.Observations.Stations.Count):
            if (indivSorted[i].errors[j] <= front[-1].errors[j]):
                front.append(indivSorted[i])

    # States.delete([ind.ID for ind in front])

    front = truncateFront(front, individuals)
    for ind in front:
        ind.pop = Supply.ModelDate.numOfPopulation
    Log.write('Front length = {0}'.format(len(front)))
    return front


def lifeCycle(coindividuals):
    modelDate, individuals1, individuals2, convErrorOld = CoevolutionData.preprocess(coindividuals)
    Supply.ModelDate.numOfPopulation = 2
    coindividuals = evolution(individuals1, individuals2, modelDate, convErrorOld)
    # postprocess(individuals)
    return copy.deepcopy(coindividuals)


def stepping():
    Data.clearData()
    Log.clear()

    individuals = []

    drag_def = 1.0
    wcr_def = 1.77e-05
    ws_def = 0.00302
    physType = 1

    Consts.State.forcingId = 0
    Consts.State.currentPeriodId = 0
    Consts.State.expId = "regularExp" + str(0) + "_" + str(Consts.State.forcingType[0]) + "_" + str(0 + 1)

    for i in range(1, 40):
        dragFunc = drag_def * i / 10
        for j in range(-5, 5, 1):
            for k in range(1, 4):
                wcr = wcr_def * pow(10, j) * k * 2.5
                ws = 0.00302
                Supply.runModel((dragFunc, physType, wcr, ws), "")  # run model and returns individual

    # runModel((drag_def, physType, wcr_def, ws_def),"")

    # return
    # Consts.State.forcingId = 0


# Consts.State.currentPeriodId = 4
# Consts.State.expId = "regularExp"+str(0)+"_"+str(Consts.State.forcingType[0])+"_"+str(0+1)

# for i in range(1, 20):
#    dragFunc = drag_def * i/10
#    for j in range(1, 32000,400):
#            wcr = wcr_def / 400 * j
#            ws = ws_def
#            runModel((dragFunc, physType, wcr, ws), "")  # run model and returns individual

# for i in range(1, 80):
#    dragFunc = drag_def * i / 10
#    for j in range(-5, 5, 1):
#        for k in range(1, 4):
#           wcr = wcr_def * pow(10, j) * k * 2.5
#            ws = 0.00302
#            runModel((dragFunc, physType, wcr, ws), "")  # run model and returns individual


def program():
    # stepping()

    # return

    # separate evolution by stations
    Data.clearData()
    Log.clear()
    # for forcingId in range(0, len(Consts.State.forcingType)):
    Consts.State.forcingId = 0  # ncep is first in co-ensemble
    for periodId in range(0, (Consts.Models.Observations.timePeriodsCount)):
        # periodId=1
        Consts.State.currentPeriodId = periodId
        # Consts.State.expId = "coevExp_time"+str(periodId)
        Consts.State.expId = "coevExp_full"
        coindividuals = []
        Log.write('Start optimization')
        Supply.ModelDate.numOfPopulation = 1
        coindividuals = lifeCycle(coindividuals)


def truncateFront(front, individuals):  # returns fron hich size is minimum from 1 to 4 and maximum sqrt(len(front))
    Log.write('Truncate Error')
    kNearest = int(math.sqrt(len(front)))
    while kNearest + 1 < len(front):  # +3 as it is a distance between any two objects
        distances = [Supply.getMultidimDistance(front[i - 1].errors, front[i].errors) for i in range(1, len(front))]
        index = distances.index(min(distances))
        # print 'Len(dist) = {0}, len(inds) = {1}, index = {2}, len(front) = {3}\n'.format(len(distances), len(individuals), index, len(front))
        if index - 1 < 0:
            del front[index + 1]
            continue
        if index + 1 == len(distances):
            del front[index]
            continue
        if getDominatingSolutions(individuals, front, index) * distances[index - 1] > getDominatingSolutions(
                individuals, front,
                index + 1) * distances[index + 1]:
            del front[index + 1]
        else:
            del front[index]
    return front


program()
