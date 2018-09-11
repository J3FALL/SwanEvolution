from datetime import datetime

from src.evo_balt import Consts


class ObservationFile:
    def __init__(self, path):
        self.path = path

    def time_series(self, from_date="", to_date=""):
        with open(self.path) as file:
            lines = self._skip_meta_info(file.readlines())
            idx_from, idx_to = self._from_and_to_idxs(lines, from_date, to_date)
            return lines[idx_from:idx_to + 1]

    def _skip_meta_info(self, lines):
        return list(filter(lambda line: line if not (line.startswith("#") or line.startswith("<")) else None, lines))

    def _from_and_to_idxs(self, lines, from_date="", to_date=""):
        idx_from, idx_to = -1, -1
        for line in lines:
            values = line.split()
            date, time = values[1], values[2]
            resulted_date = FormattedDate().target(date, time)
            if resulted_date == from_date:
                idx_from = lines.index(line)
            if resulted_date == to_date:
                idx_to = lines.index(line)

        assert idx_from < idx_to

        return idx_from, idx_to


class FormattedDate:
    def __init__(self):
        self._source_date_pattern = "%d-%m-%Y %H:%M:%S"
        self._target_date_pattern = "%Y%m%d.%H"
        self._target_suffix = "0000"

    def target(self, date, time):
        return datetime.strptime(" ".join([date, time]), self._source_date_pattern).strftime(
            self._target_date_pattern) + self._target_suffix


obs = ObservationFile("../../samples/obs/1a_waves.txt")
ts = obs.time_series(from_date=Consts.Models.Observations.timePeriodsStartTimes[Consts.State.currentPeriodId],
                     to_date=Consts.Models.Observations.timePeriodEndTimes[Consts.State.currentPeriodId])

print(len(ts))
