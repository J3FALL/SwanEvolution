class FakeModel:
    '''
    Class that imitates SWAN-model behaviour:
        it encapsulates simulation results on a model params grid:
            [drag, physics, wcr, ws] = model_output, i.e. forecasts
    '''

    def __init__(self):
        '''
        Init parameters grid
        '''

    def output(self, params):
        '''

        :param params: SWAN parameters
        :return: ForecastFile
        '''
        pass
