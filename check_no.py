import getopt, sys, os

argumentList = sys.argv[1:]
 
options = "hmo:"

long_options = ["help", "folder",]
 
try:
    arguments, values = getopt.getopt(argumentList, options, long_options)
     
    for currentArgument, currentValue in arguments:
 
        if currentArgument in ("-h", "--help"):
            print ("--folder <folder name> to check number of programs")
             
        elif currentArgument in ("-f", "--folder"):
            try:
                print (f"Displaying files inside {sys.argv[2]}: {len(os.listdir(sys.argv[2]))}")
            except NotADirectoryError as err:
                print(str(err))
             
             
except getopt.error as err:
    print (str(err))
