from math import sqrt


def error_rmse_all(forecast, observations):
    '''
    Calculate RMSE of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    result = 0.0
    for pred, obs in zip(forecast.hsig_series, observations):
        result += pow(pred - obs, 2)

    return sqrt(result / len(observations))


def error_rmse_peak(forecast, observations):
    '''
    Calculate peakwise RMSE of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    observation_peaks = [obs if obs > 1 else 1 for obs in observations]

    forcasts_peaks = [fk if fk > 1 else 1 for fk in forecast.hsig_series]

    result = 0.0
    for pred, obs in zip(forcasts_peaks, observation_peaks):
        result += pow(pred - obs, 2)

    return sqrt(result / len(observation_peaks))


def error_mae_peak(forecast, observations):
    '''
    Calculate peakwise MAE of between forecasts and observations for corresponding station
    :param forecast: Forecast object
    :param observations: ObservationFile object
    '''

    observation_peaks = [obs if obs > 1 else 1 for obs in observations]

    forcasts_peaks = [fk if fk > 1 else 1 for fk in forecast.hsig_series]

    result = 0.0
    for pred, obs in zip(forcasts_peaks, observation_peaks):
        result += abs(pred - obs)

    return result / len(observation_peaks)
