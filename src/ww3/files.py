import glob
import os
import re

PATH_TO_WW3_RESULTS = '../../samples/ww-res/'

from src.swan.files import WaveWatchObservationFile


def wave_watch_results(path_to_results=PATH_TO_WW3_RESULTS, stations=None):
    '''

    :param path_to_results: Path to directory with ww3 results stored as csv-files
    :param stations: List of stations to take
    :return: List of WaveWatchObservationFiles objects according to chosen stations
    '''

    choice = '|'.join([str(station) for station in stations])
    file_pattern = f'obs_fromww_({choice}).csv'

    files = []
    for file in glob.iglob(os.path.join(path_to_results, '*.csv')):
        if re.search(file_pattern, file):
            files.append(file)

    result = [WaveWatchObservationFile(file) for file in sorted(files)]
    return result


ww3_obs = wave_watch_results(path_to_results=PATH_TO_WW3_RESULTS, stations=[4, 5, 6])
