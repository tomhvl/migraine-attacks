
# This code converts plain .AWD datafile (output from an actigraf sensor)
# and prepares it for use with HTM studio from Numenta. The lines are
# timestamped according to the data given in the headers.

# OBS: There is some inconisistency in the headers (time HH:MM & HH:MM:SS)
# and norwegian calendar names that might need intervention.

# 03/01/2016 changed output date format to dd/mm/yyyy hh:mm
# 05/01/2016 changed dateformat to https://github.com/numenta/nupic/wiki/NuPIC-Input-Data-File-Format




import sys
import locale

from datetime import datetime, timedelta


TIME_INCREMENT = 1  # the time steps for each reading in minutes
MND_INDEX = ["placeholder", "jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]


def extractHeaderInfo(dataFile):
    """ Extract header information and return a tuple according to the
    following format (by line #):
        1 pasient/kontroll
        2 01-mar-1900 (dato)
        3 09:00 (starttid)  
        6 nr. paa aktigrafen som er brukt
        8 og nedover : sensordata
    """

    pasient = dataFile.readline().rstrip("\n")
    dato = dataFile.readline().rstrip("\r\n")
    starttid = dataFile.readline().rstrip("\r\n")
    var4 = dataFile.readline().rstrip("\n")
    var5 = dataFile.readline().rstrip("\n")
    aktigrafID = dataFile.readline().rstrip("\n")
    var7 = dataFile.readline().rstrip("\n")

    # taake only 5 first characters from starttid (some are bigger)
    starttid = starttid[:5]

    # change date from norwegian short to digit (cause of problems with locale settings win vs linux)
    dato = dato[:3] + str(MND_INDEX.index(dato[3:6])) + dato[6:]
    
    return (pasient, dato, starttid, var4, var5, aktigrafID, var7)

    

def stamp(input_name=""):
    
    # append prefix/suffix to modified datafile
    output_name = "stm_"+input_name[:-4]+".csv"
 
  
    try:
        innfile = open(input_name, 'r')
        outfile = open(output_name, 'w')
    except IOError:
        print ("Error: can\'t find file or read data.")
    else:

        # get header information for .AWD files
        head = extractHeaderInfo(innfile)
        
        #outfile.write("time,value\n")         #needed for HTM studio!
        outfile.write("timestamp,value,flag\n")
        outfile.write("datetime,int,str\n")
        outfile.write("T,\n")
        
        # get the timestamp starting value (using locale settings)
        dt = datetime.strptime(head[1]+" "+head[2], "%d-%m-%Y %H:%M")

       
        # read the data lines
        for line in innfile:

            # output datetime using: https://github.com/numenta/nupic/wiki/NuPIC-Input-Data-File-Format
            tstamp = dt.strftime("%Y-%m-%d %H:%M")

            # some edited data removed here in studio version 
            # we keep the EDIT notes in the script version
            line = line.split()
            flag = ""
              
            if len(line) > 1:
            	flag = line[1]
            	
            outfile.write("{},{:d},{}\n".format(tstamp, int(line[0]), flag))                    

            # increment datetime by 1 minute
            dt += timedelta(minutes=TIME_INCREMENT)          

        innfile.close()
        outfile.close()

        print "Done!"


        
if __name__=="__main__":
    # the locale used in current .AWD files 
    locale.setlocale(locale.LC_ALL, "no_NO")

    args = sys.argv[1]
    if args[-4:] == ".AWD":
        stamp(args)
    else:
        print "Eneter valid  (AWD) filename as argument."



    
        
