import unittest

from src.evo_balt.files import ForecastFile
from src.evo_balt.files import ObservationFile


class ObservationFileTest(unittest.TestCase):
    def test_time_series_correct(self):
        obs = ObservationFile("../samples/obs/1a_waves.txt")
        ts = obs.time_series(from_date="20140814.120000",
                             to_date="20140915.000000")
        self.assertEqual(len(ts), 253)



class ForecastFileTest(unittest.TestCase):
    def test_time_series_correct(self):
        forecast = ForecastFile("../samples/results/K1.a")
        ts = forecast.time_series()
        self.assertEqual(len(ts), 4)
