#!/usr/bin/julia -q
# Version 1.0
# Date: 13-05-15 at 03:37:06 AM IST

# Command line arguments are stored in the variable ARGS
#

filename="ami"
data_file = @sprintf("%s_data.csv",filename)
outcome_file = @sprintf("%s_y.csv",filename)


data = readdlm(data_file,',',Int64)
y = readdlm(outcome_file,',',Int64)

classTable = sort(unique(y))
#________________________________________________________________________________
 include("Bayesian.jl")
  
  
 metaData = makemetadata(data)
 (data, LUT) = encodeX(data,metaData)
 #________________________________________________________________________________
 sampleFraction = 0.01 # taking 1 percent of data for testing
 
 (trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
#________________________________________________________________________________

sampleFraction = 0.1
(trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
#________________________________________________________________________________
 include("quinticDemo.jl")
 
 sessionId = randstring(6) # Create a random string for sessionId
 sessionId = string("QUINT_",sessionId)
 println(sessionId)
 #__________________________________________
 
 (trainParams, LUT, RET_MSG) = quinticTrain(trnSet, trnOut, sessionId, LUT)
 println(RET_MSG)
 
 queryId = string(rand(1000000000:9999999999)) # generate a 10 digit random number
 
 JSONstr = quinticPredict(cvSet, classTable, sessionId, queryId, trainParams, LUT)

 # Let us write it to a file
 f = open("quinticJSON.txt", "w")
 write(f, JSONstr)
 exit()
 

