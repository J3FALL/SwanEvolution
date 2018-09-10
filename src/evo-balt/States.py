import Consts
import Log
import Supply
import os
import re
import shutil


def copy(new_name, name_postfix, path=Consts.Models.SWAN.pathToStatesRoot):
    Log.write('New state is copied')
    if not os.path.exists(path + '\\STATES\\'):
        os.makedirs(path + '\\STATES\\')
    name = '{0:04d}{1:02d}{2:02d}{3:02d}'.format(Supply.ModelDate.Year, Supply.ModelDate.Month,
                                                 Supply.ModelDate.Days, Supply.ModelDate.Hours)
    shutil.copy(path + str(name) + str(name_postfix), path + '\\STATES\\')
    os.rename(path + '\\STATES\\' + str(name) + str(name_postfix),
              path + '\\STATES\\' + str(new_name) + str(name_postfix))
    return 0


def delete(indexes, path=Consts.Models.SWAN.pathToStatesFolder):
    Log.write('Old states are being deleted.')
    # if (os.path.exists(path)):
    for i, name in enumerate(os.listdir(path)):
        # if int(name.split('_')[0]) not in indexes:
        try:
            # if i%2 == 0:
            #    Log.write('{0} is not in indexes {1}'.format(name.split('_')[0], indexes))
            os.remove(path + name)
        except OSError:
            Log.write('Error: {0} does not exist.'.format(name.split('_')[0]))
            pass
    return 0


def average(indexes, postfix, path=Consts.Models.SWAN.pathToStatesFolder):
    Log.write('State averaging')
    delete(indexes)
    list_of_state_files = [read_state(str(file.split('_')[0]), postfix, path) for file in os.listdir(path)]

    if (len(list_of_state_files) > 0):
        for i, state_file in enumerate(list_of_state_files):
            if i == 0:
                continue
            for j, line in enumerate(state_file):
                list_of_state_files[0][j] = map(lambda state_0_j, state_i_j: state_0_j + state_i_j,
                                                list_of_state_files[0][j], line)

        return [map(lambda l: l / len(list_of_state_files), line) for line in list_of_state_files[0]]
    else:
        return []


def read_state(name, name_postfix, path):
    Log.write('State {} is being readen'.format(name + name_postfix))
    with open(path + name + name_postfix) as state_file:
        content = state_file.readlines()

    states = [map(lambda num: float(num), re.findall(r"[-+]?\d*\.\d+|\d+", c)) for c in content]
    return states


def write_state(state, path, name, name_postfix):
    Log.write('Averaged state written')
    with open(path + name + name_postfix, "w") as text_file:
        for line in state:
            line_string = ''
            for value in line:
                line_string += '{0:7.3f}'.format(value)

            text_file.write(line_string + '\n')
    return 0
