import Supply
import math


def getRange(y, m, d, h, **options):
    dateStr = Supply.getDate(y, m, d, h, 0)
    with open("D:\\Balt-P\\Balt-P-Shell-1.5.2\\Grids\\mask2m.txt", 'r') as Mfile:
        Mcontent = list(Mfile)
    with open("D:\\Balt-P\\Balt-P-Shell-1.5.2\\work\\HIRLAM\\UV\\UV02_" + dateStr + ".txt", 'r') as UVfile:
        UVcontent = list(UVfile)
    u = []
    v = []
    # meteoForecastTime = (int)(Supply.readIniFile('D:\Balt-P\Balt-P-Shell-1.5.2\config.ini', 'SimulationParams', 'meteoforecasttime'))
    num_of_fields_in_UV = 2 * ((int)(
        Supply.readIniFile('D:\Balt-P\Balt-P-Shell-1.5.2\config.ini', 'SimulationParams', 'meteoforecasttime')) /
                               int(Supply.readIniFile('D:\Balt-P\Balt-P-Shell-1.5.2\config.ini', 'SimulationParams',
                                                      'meteoforecaststep')) + 1)

    for hour in xrange(num_of_fields_in_UV):
        for str in xrange(options['min_y'], options['max_y']):
            bufUV = map(lambda y: float(y), filter(lambda x: x != '', [(val.replace('\n', '')) for val in
                                                                       UVcontent[375 * hour + str].split(' ')]))
            bufM = [int(val.replace('\n', '')) for val in Mcontent[str].split(
                ' ')]  # re.split(r' \\t  \\n', UVcontent[str])#Mcontent[str].split(' ')# do not need to split, 'cause it is already split into list. Where each element is either 0-1 or empty space
            for col in xrange(options['min_x'], len(bufM)):
                if (int(bufM[col] != 0)):
                    if (hour % 2 == 0):
                        u.append(float(bufUV[col]))
                    else:
                        v.append(float(bufUV[col]))

    wind = map(lambda x, y: math.sqrt(x ** 2 + y ** 2), u, v)
    '''print min(wind), max(wind)
    print int(min(wind)/4), int(max(wind)/4)
    print round(min(wind)) , round(max(wind))'''
    return [Round(float(min(wind))), Round(float(max(
        wind)))]  # this function returns values depicting interval correlated with distribution of drag function's values,
    # hence if the origin value of wind is closer to one's border according to the function division, (2.1 -> 4 or 8.4 -> 8)
    # then it belongs to specified interval. For better explanation, examples and understanding uncomment print-functions.


def Round(num):
    if (num / 4 - int(num / 4) >= 0.5):
        return int(num / 4) + 1
    else:
        return int(num / 4)
