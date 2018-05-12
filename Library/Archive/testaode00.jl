#!/usr/bin/julia -q
# Version 0.0
# Date: 18-05-15 at 06:54:28 AM IST
# Version 4.0
# Date: 10-05-15 at 11:00:00 PM IST

# Command line arguments are stored in the variable ARGS
#

include("Library/general.jl")
include("Library/Bayesian.jl")


filename="ami"

println("Reading dataset: $filename")

data_file = @sprintf("%s_data.jla",filename)
outcome_file = @sprintf("%s_y.jal",filename)

data = readArray(data_file)
y = readArray(outcome_file)



# metaData needs to be created
metaData = makemetadata(data)
(data, LUT) = encodeX(data,metaData)

(threshold, stdval) = getThreshold(data,y,LUT)

println("Iterating ")
maxIter = 500
sampleFraction = 0.1
TPR = zeros(Float64, maxIter)
TNR = zeros(Float64, maxIter)
uniqY = sort(unique(y))

trueRatio = 0.5
dataSize = 15
(newData, newOut) = preboost(data, y, 0.5,iround(dataSize*size(data,1)))
for iter =1:maxIter
	@printf("Iteration (%d): %d\r",maxIter,iter)
	(trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
	(yLogProbT, nodeLogProbT) =	trainAODE(newData,newOut,LUT)
	(classIndex ,_ ,_) = predictAODE(cvSet, yLogProbT, nodeLogProbT)
	obs = uniqY[classIndex]
	(TPR[iter], TNR[iter], _) = performanceStats(cvOut,obs)

end

println()
meanTPR = mean(TPR)
meanTNR = mean(TNR)
@printf("%f\t%f\n", meanTPR, meanTNR)


println("Ready to plot")
using Gadfly

pic = plot(Theme(background_color=color("white")),
			layer(x=[1:maxIter], y=TPR, Geom.line, 
				Geom.hline(color="green"), 
				yintercept=[meanTPR],
				Theme(default_color=color("#A1CAF1")) ),
		layer(x=[1:maxIter], y=TNR, Geom.line, 
				Geom.hline(color="red"), 
				yintercept=[meanTNR],
				Theme(default_color=color("#FE6F5E")) )
	)

draw(PNG("preboost-AODE.png", 24cm, 15cm), pic)


