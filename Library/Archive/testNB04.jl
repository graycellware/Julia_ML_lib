#!/usr/bin/julia -q
# Version 4.0
# Date: 10-05-15 at 11:00:00 PM IST

# Command line arguments are stored in the variable ARGS
#

if (length(ARGS) == 0)
	println("Usage: testNB.jl <.mat file>")
	exit(-1)
end


println("Loading Libraries ...")
ERR_FLAG = false
using Gadfly
include("Bayes.jl")


println("Reading data ...")
fileparts = match(r"^\s*(\S+?)\.(.+?)$", ARGS[1])
if (fileparts == nothing)
	ERR_FLAG = true
	continue
end
filename=fileparts.captures[1]
data_file = @sprintf("%s_data.csv",filename)
outcome_file = @sprintf("%s_y.csv",filename)
metadata_file = "metaData.csv"

data = readdlm(data_file,',',Int64)
y = readdlm(outcome_file,',',Int64)


# metaData needs to be created
metaData = makemetadata(data)
(data, LUT) = encodeX(data,metaData)

(threshold, stdval) = getThreshold(data,y,LUT)

println("Iterating ")
maxIter = 1000
sampleFraction = 0.1
TPR = zeros(Float64, maxIter)
TNR = zeros(Float64, maxIter)
uniqY = sort(unique(y))

(newData, newOut) = prevalence(data, y, 0.5,(40*size(data,1)))
for iter =1:maxIter
	(trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
	(yLogProbT, nodeLogProbT) =	trainNB(newData,newOut,LUT)
	(classIndex ,_ ,_) = predictNB(cvSet, yLogProbT, nodeLogProbT)
	obs = uniqY[classIndex]
	(TPR[iter], TNR[iter], _) = performanceStats(cvOut,obs)
	
end

meanTPR = mean(TPR)
meanTNR = mean(TNR)
@printf("%f\t%f\n", meanTPR, meanTNR)


println("Ready to plot")


pic = plot(Theme(background_color=color("white")),
layer(x=[1:maxIter], y=TPR, Geom.line, 
		Geom.hline(color="green"), yintercept=[meanTPR],
		Theme(default_color=color("#A1CAF1")) ),
layer(x=[1:maxIter], y=TNR, Geom.line, 
		Geom.hline(color="red"), yintercept=[meanTNR],
		Theme(default_color=color("#FE6F5E")) )
	)

draw(PNG("myplot_50_40_1k.png", 12cm, 6cm), pic)


