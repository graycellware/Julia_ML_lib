#!/usr/bin/julia -q
# Version 5.0
# Date: 15-05-15 at 09:43:50 AM IST

# Command line arguments are stored in the variable ARGS
#

if (length(ARGS) == 0)
	println("Usage: testNB.jl <.mat file>")
	exit(-1)
end


println("Loading Libraries ...")
ERR_FLAG = false


include("Library/general.jl")
include("Library/Bayesian.jl")


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


sampleFraction = 0.1

uniqY = sort(unique(y))
#______________________________________
ratio = 0.5
newSize = 50

(trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
(newData, newOut) = preboost(trnSet, trnOut, ratio, (newSize*size(data,1)))

#______________________________________
include("testDependence.jl")

numNodes = size(LUT,1)
numCols = maximum(LUT[:,3])
incidenceMatrix = zeros(Int64,numCols, numCols)

(N, E) = trainAODE(newData,newOut,LUT) # This to get the edgeLogProbs only
incidenceMatrix = colDependence(E,LUT)


graphFile = "graphView.tex"
of = open(graphFile, "w")

Header = """\\documentclass[tikz,margin=5pt]{standalone}
\\usetikzlibrary{graphs,graphdrawing,arrows}
\\usegdlibrary{force}
\\begin{document}
\\tikz[spring layout, node distance=25mm,>=latex']{
"""
write(of,Header)

for k = 1:numCols
	write(of, @sprintf("\\node (%d) {column\\_%s};\n",k,string(k)))
end
write(of, "\\draw\n")

for k = 1:(numCols-1), l = (k+1):numCols
	if(incidenceMatrix[k,l] == 1)
		write(of, @sprintf("(%d) edge (%d)\n", k, l))
	end
end
write(of,";\n")
write(of,"}\n\\end{document}")
close(of)






	

