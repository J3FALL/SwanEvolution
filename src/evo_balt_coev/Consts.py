class Debug(object):
    debugDragFunc = 0.05
    debugMode = True
    runModel = True and not debugMode


class State:
    currentStationId = 0
    separateStationsMode = False
    currentPeriodId = 0
    expId = "GenExp"

    forcingId = 0
    forcingType = ["ncep", "era"]

    readInitialFromFile = False
    initialFolder = "D:\\GeneticExp\\new-times\\"


class Models(object):
    class SWAN(object):
        @staticmethod
        def configName():
            return "2014_" + State.forcingType[State.forcingId]

        @staticmethod
        def pathToExec():
            return "D:\Karsky_SWAN\Stereo\swanrun.bat " + Models.SWAN.configName()

        @staticmethod
        def pathToConfig():
            return "D:\Karsky_SWAN\Stereo\\" + Models.SWAN.configName() + ".swn"

        pathToFolder = "D:\Karsky_SWAN\Stereo"
        pathToExpNotesFolder = "D:\GeneticExp\\"
        pathToStatesRoot = "D:\Karsky_SWAN\Stereo\SWAN_STATES\\"
        pathToStatesFolder = "D:\Karsky_SWAN\Stereo\SWAN_STATES\STATES\\"
        pathToResults = "D:\\Karsky_SWAN\\results\\"

        class OutputColumns(object):
            Hsig = 2

        class Stations(object):
            FullNames = ["K1a.tab", "K2a.tab", "K3a.tab"]

            @staticmethod
            def getNames():
                if (State.separateStationsMode):
                    return [Models.SWAN.Stations.FullNames[State.currentStationId]]
                else:
                    return Models.SWAN.Stations.FullNames

        class Parameters(object):
            class WindCoeff(object):
                name = "READINP WIND"
                valueColumnId = 2
                defaultValue = 1.0
                indInParamsArray = 0
                first = [0.03896535, 3.2077377, 0.05730960, 0.09018184, 0.14470722, 5.6515009, 0.07234299, 0.09359072]
                first2 = [0.03896535, 3.2077377, 0.05730960, 0.09018184, 0.14470722, 5.6515009, 0.07234299, 0.09359072]

                # first2 = [0.45, 1.1, 1.73, 1.3, 0.5, 0.2, 0.9, 0.5]

            class PhysicsType:
                name = "GEN"
                typesNames = ["GEN1", "GEN3"]
                valueFromArray = True
                defaultValue = 1
                valueColumnId = 0
                indInParamsArray = 1

            class ensCoeff:
                name = "ensCoeff"
                valueFromArray = True
                defaultValue = 0.5
                indInParamsArray = 4
                # first = [0.5,0.2,0.7,1.4,2.0,0.1,0.6,0.8]
                # first2 = [0.35, 0.12, 0.87, 0.4, 1.1, 2.1, 0.2, 0.08]
                first = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                first2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

            class WhiteCappingRate:
                name = "cds2="
                defaultValue = 0.0000236
                valueColumnId = 1
                indInParamsArray = 2
                addNameToValue = True
                first = [4.063645e-09, 3.462037e-02, 2.805453e-15, 8.622384e-07, 1.947672e-09, 1.312024e-01,
                         4.227292e-02, 2.535043e-01]
                first2 = [4.063645e-09, 3.462037e-02, 2.805453e-15, 8.622384e-07, 1.947672e-09, 1.312024e-01,
                          4.227292e-02, 2.535043e-01]
                # first2 = [4.9e-06,3.9e-05,1.8e-05,3.1e-06,7.2e-05,0.3e-05,2.25e-05,1.15e-05]

            class WaveSteepness:
                name = "stpm="
                defaultValue = 0.00302
                valueColumnId = 2
                indInParamsArray = 3
                addNameToValue = True
                first = [0.003777891, 0.005810400, 0.001115458, 0.003174824, 0.004787767, 0.004295600, 0.001703456,
                         0.002556799]

    class Observations(object):
        pathToFolder = "D:\Karsky_SWAN\obs\\"

        class Stations(object):
            FullNames = ["1a_waves.txt", "2a_waves.txt", "3a_waves.txt"]
            FullCount = len(FullNames)
            Count = FullCount

            if (State.separateStationsMode):
                Count = 1

            @staticmethod
            def getNames():
                if (State.separateStationsMode):
                    return [Models.Observations.Stations.FullNames[State.currentStationId]]
                else:
                    return Models.Observations.Stations.FullNames

            @staticmethod
            def getAllNames():
                Models.Observations.Stations.FullNames

        class OutputColumns(object):
            Hsig = 4

        # @staticmethod
        # def getTimeStepsCount(id):
        #    return (Models.Observations.timePeriodEnds[id] - Models.Observations.timePeriodStarts[id] + 1)

        # timePeriodStarts = [0, 40, 251]
        # timePeriodEnds = [40,80,268]

        # timePeriodsStartTimes = ["20140814.120000", "20140819.120000", "20140914.210000"]
        # timePeriodEndTimes = ["20140819.120000", "20140824.120000", "20140918.030000"]

        # timePeriodsStartTimes = ["20140814.120000", "20140822.000000", "20140831.090000", "20140902.060000", "20140912.120000"]
        # timePeriodEndTimes = ["20140817.150000", "20140828.210000", "20140902.060000", "20140906.150000", "20140915.000000"]
        timePeriodsStartTimes = ["20140814.120000"]
        timePeriodEndTimes = ["20140915.000000"]
        timePeriodsCount = 1


class Evolution(object):
    criticalConvergenceExtent = 16
    errorBorderForExtent = 0.03
    errorBorderForFinish = 0.01
    convergenceExtent = 0
    populationMaxSize = 6
    minPopulationsNumber = 4
    maxPopulationsNumber = 16


class individual:
    def __init__(self, ID, pop, errors, fullErrors, params, forecasts):
        self.ID = ID
        self.pop = pop  # ulation

        self.params = params
        self.dragFunc = params[Models.SWAN.Parameters.WindCoeff.indInParamsArray]
        self.physicsType = params[Models.SWAN.Parameters.PhysicsType.indInParamsArray]
        self.wcr = params[Models.SWAN.Parameters.WhiteCappingRate.indInParamsArray]
        self.ws = params[Models.SWAN.Parameters.WaveSteepness.indInParamsArray]

        self.ensCoeff = params[Models.SWAN.Parameters.ensCoeff.indInParamsArray]

        self.errors = errors

        self.fullErrors = fullErrors

        self.forecasts = forecasts
