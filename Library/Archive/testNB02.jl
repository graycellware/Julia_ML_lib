#!/usr/bin/julia -q
# Version 2.0
# Date: 09-05-15 at 08:38:41 PM IST

# Command line arguments are stored in the variable ARGS
#
ERR_FLAG = false
include("Bayes.jl")

if (length(ARGS) == 0)
	println("Usage: testNB.jl <.mat file>")
end

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
data, LUT = encodeX(data,metaData)

yLogProbs, nodeLogProbs = trainNB(data, y, LUT)

writedlm("yLogProbs.csv", yLogProbs, ',' )
writedlm("nodeLogProbs.csv", nodeLogProbs, ',' )
writedlm("LUT.csv", LUT, ',' )
