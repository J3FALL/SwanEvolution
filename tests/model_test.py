import unittest

from src.evo_old.model import FakeModel
from src.evo_old.model import GridFile


class FakeModelTest(unittest.TestCase):
    def test_init_observations_correct(self):
        model = FakeModel(grid_file=GridFile("../samples/grid_era_full.csv"))

        self.assertEqual(3, len(model.observations))

        for station in model.observations:
            self.assertEqual(253, len(station))
