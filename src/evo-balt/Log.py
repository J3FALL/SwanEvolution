import datetime
import os
import Consts
def write(s):
    with open(Consts.Models.SWAN.pathToExpNotesFolder+str(Consts.State.expId)+'_logfile.txt', 'a') as text_file:
        text_file.write("%s     %s\n" % (datetime.datetime.now().strftime("%H:%M:%S on %B.%d.%Y"), s))

def clear():
    try:
        os.remove(Consts.Models.SWAN.pathToExpNotesFolder+str(Consts.State.expId)+'_logfile.txt' )
    except OSError:
        print("No log file")