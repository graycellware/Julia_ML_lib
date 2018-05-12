#!/usr/bin/julia -q
# Version 5.0
# Date: 13-05-15 at 01:43:12 AM IST

# Command line arguments are stored in the variable ARGS
#

if (length(ARGS) == 0)
	println("Usage: testNB.jl <.mat file>")
	exit(-1)
end


println("Loading Libraries ...")
ERR_FLAG = false

include("Bayesian.jl")


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
#______________________________________________________________________
println("Iterating ")
maxIter = 1
sampleFraction = 0.1
TPR = zeros(Float64, maxIter)
TNR = zeros(Float64, maxIter)
uniqY = sort(unique(y))
#______________________________________
ratio = 0.5
newSize = 50
(newData, newOut) = preboost(data, y, ratio, (newSize*size(data,1)))


for iter =1:maxIter
	@printf("Iteration (%d): %d\r",maxIter,iter)
	(trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
	(N, E) =	trainAODE(newData,newOut,LUT) # This to get the edgeLogProbs only
	
	(Y,N) = getNBparamsAODE(E)
	
	(classIndex ,predProb ,predVal) = predictNB(cvSet, Y,N)
	predClass = uniqY[classIndex]

	# Testing getCauses ...
	include("Causes.jl")
	# take a random candidate from the cvSet
	candSize = size(cvOut,1)
	selected = randperm(candSize)[1]
	(nodeCause, edgeCause) = getCauses(E,predClass[selected],cvSet[selected,:],LUT)
	

	if (predClass[selected] == 1)
				
		if(predProb[selected] > 0.8)
			println("Risk Level: Very High")
		elseif (predProb[selected] > 0.7)
			println("Risk Level: High")
		elseif (predProb[selected] > 0.6)
			println("Risk Level: Moderate")
		else
			println("Risk Level: Borderline High")
		end
	else
		if (predProb[selected] > 0.4)
			println("Readmission Risk: Borderline Low")
		else
			println("Readmission Risk: Low")
		end
	end
	
	if (predClass[selected] == 1)
		println("Reasons:")
	else
		println("Concerns:")
	end
		
		
	if nodeCause != nothing
		numel = size(nodeCause,1)
		println("Feature values:")
		for k = 1:numel
			@printf("\t%d) (Column: %d, Value: %d) Prob= %6.4f\n", 
				k, nodeCause[k,1],nodeCause[k,2],nodeCause[k,3], )
		end
	end
	if edgeCause != nothing
		numel = size(edgeCause,1)
		println("Feature Combinations (indicative of Readmissions):")
		for k = 1:numel
			@printf("\t%d) (Column: %d, Value: %d) (Column: %d, Value: %d) Prob= %6.4f\n", 
					k, edgeCause[k,1],edgeCause[k,2],
					edgeCause[k,3],edgeCause[k,4],
					edgeCause[k,5] )
		end
	end
	
	#(TPR[iter], TNR[iter], _) = performanceStats(cvOut,predClass)
	
end



#=
meanTPR = mean(TPR)
meanTNR = mean(TNR)
println()
@printf("%f\t%f\n", meanTPR, meanTNR)


#______________________________________________________________________

println("Ready to plot")

using Gadfly

pic = plot(Theme(background_color=color("white")),
layer(x=[1:maxIter], y=TPR, Geom.line, 
		Geom.hline(color="green"), yintercept=[meanTPR],
		Theme(default_color=color("#A1CAF1")) ),
layer(x=[1:maxIter], y=TNR, Geom.line, 
		Geom.hline(color="red"), yintercept=[meanTNR],
		Theme(default_color=color("#FE6F5E")) )
	)

printstr = @sprintf("AODE_%s_%s_%s.png",filename, ratio,newSize)
draw(PNG(printstr, 12cm, 6cm), pic)
=#

