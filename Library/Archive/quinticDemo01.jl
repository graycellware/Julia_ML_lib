
# Version 1.0
# Date: 13-05-15 at 01:43:07 AM IST
include("Causes.jl")
#----------------------------------------------------------------------
#---------------------------------------------------------------------
function getPredJSONstr(	sessionId::ASCIIString, 
							queryId::ASCIIString, 
							candData::Array{Any,2})
 
 #-----------------------------------------
 # candData = cell(4, dataSize)
 #-----------------------------------------
 # Cell Structure
 # --------------
 # candData[1,*]: predicted class
 # candData[2,*]: predicted probability
 # candData[3,*]: nodeCause (see below)
 # candData[4,*]: edgeCause (see below)
 #-----------------------------------------
 
 # Create a set of functions
 # Session block
 S(x,y) = (@sprintf("{\"Session ID\": %s, \"Session\": [\n %s]}",x,y))
 
 # Query block
 Q(x,y) = (@sprintf("{\"Query ID\": %s, \"Query\": [\n %s]}",x,y))
 
 # Candidate block: a= Candidate index in data; b= predClass; c= predProb; d=report
 C(a,b,c,d) = (	 @sprintf("{\"Candidate Index\": \"%s\", 
 						\"Predicted Class\": \"%s\",
 						\"Predicted Probability\": \"%s\",
 						\" Report\": [\n %s]}\n",a,b,c,d))
 
 # Report block:
 R(x,y) = (@sprintf("{\"NODE\": [\n %s]},{\"EDGE\": [\n %s]}",x,y))
 
 # Node block
 N(x,y,z) = (@sprintf("{\"column No.\": \"%s\", \"Value\": \"%s\", 
 						\"Probability\": \"%s\"}\n",x,y,z))
 # Edge block
 E(a,b,c,d,x) = (@sprintf("{\"column No.\": \"%s\", \"Value\": \"%s\",
 							\"column No.\": \"%s\", \"Value\": \"%s\",  
 							\"Probability\": \"%s\"}\n",
 							 a,b,c,d,x))
 							 
 #--------------------------------------------------------------
 
 # Let us iterate through each candidate
 num_cands = size(candData,2)
 
 CandStr = ""
 for k = 1:num_cand
 	predClass = candData[1,k]
 	predProb =  candData[1,k]
 	nodeArray = candData[1,k]
 	edgeArray = candData[1,k]
 	
 	
 	Nstr = "" # empty string
 	# nodeArray could potentially be empty
 	if(nodeArray != nothing)
 		numNodes = size(nodeArray,1)
 		for j = 1: numNodes
 			tmpstr = N(nodeArray[1,j], nodeArray[2.j], nodeArray[3,j])
 			Nstr = string(Nstr, ',', tmpstr)
 		end
 	end
 	
 	
 	
 	Estr = ""
 	# edgeArray is seldom empty, but still check ...
 	if(edgeArray != nothing)
 		numEdges = size(edgeArray,1)
 		for j = 1: numEdges
 			tmpstr = E(edgeArray[1,j], edgeArray[2.j], edgeArray[3,j], 
 						edgeArray[3,j],edgeArray[3,j])
 			Estr = string(Estr, ',', tmpstr)		
 		end
 	end
 	
 	# Concatenate these two to form a report
 	repstr = R(Nstr,Estr)
 	
 	# Concatenate report to Candidate
 	candstr = C(string(k),
 				string(predClass),
 				string(predProb),
 				repstr))
 	#  Finally append candidate to other candidates
 	CandStr = string(CandStr,', ',candstr)
 end
 quesryStr = Q(queryId,CandStr)
 return S(sessionId,queryStr)
end

#----------------------------------------------------------------------
#---------------------------------------------------------------------
function quinticTrain(data::Array{Int64}, outcomes::Array{Int64},
		sessionId::AbstractString)


 RET_MSG = ""
 ratio = 0.5
 newSize = 50
 include("Bayesian.jl")
 uniqY = sort(unique(y))
 metaData = makemetadata(data)
 (data, LUT) = encodeX(data,metaData)
 (newData, newOut) = preboost(data, outcomes, ratio, (newSize*size(data,1)))
 (trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
 (_, edgeLogProbs) =	trainAODE(newData,newOut,LUT) # This to get the edgeLogProbs only
	#= To Do: Implement using LMDB.jl
 	See http://wildart.github.io/LMDB.jl/manual/
	 =#
 try
	outputFileData = @sprintf("%s_params.csv",sessionId)
	outputFileLUT = @sprintf("%s_LUT.csv",sessionId)
	writedlm(outputFileData,edgeLogProbs,',')
	writedlm(outputFileLUT,LUT,',')
 catch
 	return @sprintf("ERROR: create %s*.csv", sessionId)
 end
 return @sprintf("SUCCESS: Parameters stored in %s*.csv files", sessionId)
end
#----------------------------------------------------------------------
#---------------------------------------------------------------------
function quinticPredict!(data::Array{Int64}, sessionId::ASCIIString, queryId::ASCIIString)

 # Read parameters needed for prediction ...
 
 #= To Do: Implement using LMDB.jl
 See http://wildart.github.io/LMDB.jl/manual/
 =#

 try
 	outputFileData = @sprintf("%s_params.csv",sessionId)
 	outputFileLUT = @sprintf("%s_LUT.csv",sessionId)
 	readdlm(outputFileData,edgeLogProbs,',')
 	readdlm(outputFileLUT,LUT,',')
 catch
 	return @sprintf("ERROR:%s: Read %s*.csv", queryId, sessionId)
 end

 dataSize = size(data,1)
 candData = cell(4, dataSize)
 #-----------------------------------------
 # candData = cell(4, dataSize)
 #-----------------------------------------
 # Cell Structure
 # --------------
 # candData[1,*]: predicted class
 # candData[2,*]: predicted probability
 # candData[3,*]: nodeCause (see below)
 # candData[4,*]: edgeCause (see below)
 #-----------------------------------------
 
  
 # Get ready to predict ...
 # Derive naive Bayes parameters 
 (yLogProbs,nodeLogProbs) = getNBparamsAODE(edgeLogProbs)
 
 # predict in a single shot ...
 (classIndex, predProb, predVal) = predictNB(cvSet, yLogProbs,predProb)
 predClass = uniqY[classIndex]
 
 # Process the outcome of each candidate to create a JSON string
 for selected =1:dataSize
 	
 	candData[1, selected] = predClass[selected]
 	candData[2, selected] = predProb[selected]
 	
 	#(nodeCause, edgeCause) = getCauses(edgeLogProbs,predClass[selected],
 	#									cvSet[selected,:],LUT)
 	
 	(candData[3, selected], candData[4, selected]) =
 		getCauses(edgeLogProbs,predClass[selected], cvSet[selected,:],LUT)
 end
 
 # Now proceed to generate JSON string ...
 return JSONstr = getPredJSONstr(sessionId, queryId, candData)
end
 
