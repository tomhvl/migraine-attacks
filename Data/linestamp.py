
# This code converts plain .AWD datafile (output from an actigraf sensor)
# and prepares it for use with HTM studio from Numenta. The lines are
# timestamped according to the data given in the headers.

# OBS: There is some inconisistency in the headers (time HH:MM & HH:MM:SS)
# and norwegian calendar names that might need intervention.


import sys
import locale

from datetime import datetime, timedelta


TIME_INCREMENT = 1  # the time steps for each reading in minutes
MND_INDEX = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]


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
    dato = dataFile.readline().rstrip("\n")
    starttid = dataFile.readline().rstrip("\n")
    var4 = dataFile.readline().rstrip("\n")
    var5 = dataFile.readline().rstrip("\n")
    aktigrafID = dataFile.readline().rstrip("\n")
    var7 = dataFile.readline().rstrip("\n")

    # taake only 5 first characters from starttid (some are bigger)
    starttid = starttid[:5]

    # change date from norwegian short to digit
    #dato = dato[:3] + MND_INDEX.index(dato[4:7]) + dato[6:]
    
    return (pasient, dato, starttid, var4, var5, aktigrafID, var7)

    

def stamp(input_name=""):
    
    # append prefix/suffix to modified datafile
    output_name = "stamped_"+input_name+".csv"
 
  
    try:
        innfile = open(input_name, 'r')
        outfile = open(output_name, 'w')
    except IOError:
        print ("Error: can\'t find file or read data.")
    else:

        # get header information for .AWD files
        head = extractHeaderInfo(innfile)
        
        outfile.write("time,value\n")         #needed for HTM studio!
        
        # get the timestamp starting value (using locale settings)
        dt = datetime.strptime(head[1]+" "+head[2], "%d-%b-%Y %H:%M")

        # decides how to categorize the lines (optional)
        # this would be done to create virtual cycles / group time periods
        stamp = 0
        stamp_step = 24 * 60
        counter = 0
        
        # read the data lines
        for line in innfile:

            # output datetime using:  MM/DD/YYYY HH:mm
            tstamp = dt.strftime("%m/%d/%Y %H:%M")

            # outfile.write(
            line = line.split()[0]  # some edited data removed here!           
            outfile.write("{}, {:d}\n".format(tstamp, int(line)))                    

            # increment datetime by 1 minute
            dt += timedelta(minutes=TIME_INCREMENT)

            # increase the line counter and cycle if needed
            counter = (counter + 1) #% stamp_step
            stamp =  counter # (counter // 60)
            

        innfile.close()
        outfile.close()

        print "Done!"


        
if __name__=="__main__":
    # the locale used in current .AWD files 
    locale.setlocale(locale.LC_ALL, "nor_nor")

    args = sys.argv[1]
    if args[-4:] == ".AWD":
        stamp(args)
    else:
        print "Eneter valid  (AWD) filename as argument."



    
        
