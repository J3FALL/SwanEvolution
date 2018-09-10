import fileinput
import os
import subprocess
import time

import numpy as np

import Consts
import Log


def readSwanConfig(fileName, option, argNums):
    valueString = ""
    file = open(fileName, "r")
    text = file.read()

    lines = text.split('\n')
    if (any(option in line for line in lines)):
        foundLine = [line for line in lines if option in line][0]
        linesParts = np.asarray(foundLine.split())
        result = linesParts[argNums]
        if (len(argNums) == 1):
            return result[0]
        else:
            return result
    '''
    if (type=="option"):
        m = re.search(option+'=(.+?)\n',text)
        if m:
            valueString = m.group(1)
    if (type="date"):
        m = re.search(option + ' (.+?)   '\n', text)
        if m:
    valueString = m.group(1)
    '''

    return valueString


def writeSwanConfig(fileName, option, argNums, value):
    # f = fileinput.input(fileName, inplace=1, backup='.bak'  )
    if (not Consts.Debug.debugMode):
        time.sleep(1.0)

        Log.write("Start write to confg")
        for line in fileinput.input(fileName, inplace=1, backup='.bak'):
            newLine = line
            if (option in line):
                linesParts = np.asarray(line.split())
                if (len(argNums) == 1):
                    forReplace = linesParts[argNums[0]]
                    line = line.replace(forReplace, value)
                else:
                    argInd = 0
                    for argNum in argNums:
                        # forReplace = linesParts)[argNum]
                        # line=line.replace(forReplace, list(reversed(value))[argInd])
                        linesParts[argNum] = value[argInd]
                        line = " ".join(linesParts)
                        argInd += 1
                    line += "\n"
            print
            line,


def runSwanBatFile():
    if (Consts.Debug.runModel):
        savedWorkDir = os.getcwd()
        os.chdir(Consts.Models.SWAN.pathToFolder)

        p = subprocess.Popen(Consts.Models.SWAN.pathToExec())
        os.chdir(savedWorkDir)
        p.wait()
