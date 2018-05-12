
# Version 3.0
# Date: 15-05-15 at 02:08:21 AM IST
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
 S(x,y) = (@sprintf("{\"Session ID\": %s, \"Session\": [\n %s]\n}\n",x,y))
 
 # Query block
 Q(x,y) = (@sprintf("{\"Query ID\": %s, \"Query\": [\n %s]\n}\n",x,y))
 
 # Candidate block: a= Candidate index in data; b= predClass; c= predProb; d=report
 C(a,b,c,d) = (	 @sprintf("{\"Candidate Index\": \"%s\", 
 						\"Predicted Class\": \"%s\",
 						\"Predicted Probability\": \"%s\",
 						\" Report\": [\n %s]}\n",a,b,c,d))
 
 # Report block:
 R(x,y) = (@sprintf("{\"NODE\": [\n %s]},{\"EDGE\": [\n %s]\n}\n",x,y))
 
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
 num_cand = size(candData,2)
 
 CandStr = ""
 for k = 1:num_cand
 	predClass = candData[1,k]
 	predProb =  candData[2,k]
 	nodeArray = candData[3,k]
 	edgeArray = candData[4,k]
 	
 	
 	Nstr = "" # empty string
 	# nodeArray could potentially be empty
 	if((nodeArray != nothing) && (!isempty(nodeArray)))
 		numNodes = size(nodeArray,1)
 		for j = 1: numNodes
 			tmpstr = N(string(nodeArray[j,1]),
 					string(nodeArray[j,2]), 
 					string(nodeArray[j,3]))
 					
 			Nstr = string(Nstr, ", ", tmpstr)
 		end
 	else
 			tmpstr = N("NULL", "NULL", "NULL")
 					
 			Nstr = string(Nstr, ", ", tmpstr)
 	end
 	
 	
 	
 	Estr = ""
 	# edgeArray is seldom empty, but still check ...
 	if((edgeArray != nothing) && (!isempty(edgeArray)))
 		numEdges = size(edgeArray,1)
 		for j = 1: numEdges
 			tmpstr = E( string(edgeArray[j,1]), 
 						string(edgeArray[j,2]), 
 						string(edgeArray[j,3]), 
 						string(edgeArray[j,4]),
 						string(edgeArray[j,5]))
 						
 			Estr = string(Estr, ", ", tmpstr)		
 		end
 	end
 	
 	# Concatenate these two to form a report
 	repstr = R(Nstr,Estr)
 	
 	# Concatenate report to Candidate
 	candstr = C(string(k),
 				string(predClass),
 				string(predProb),
 				repstr)
 	#  Finally append candidate to other candidates
 	CandStr = string(CandStr, ", ", candstr)
 end
 queryStr = Q(queryId,CandStr)
 return S(sessionId,queryStr)
end

#----------------------------------------------------------------------
#---------------------------------------------------------------------
function quinticTrain(data::Array{Int64}, outcomes::Array{Int64},
		sessionId::ASCIIString, LUT::Array{Int64})

 ratio = 0.5
 newSize = 50
 
 RET_MSG = @sprintf("SUCCESS: Parameters stored in %s*.csv files", sessionId)
 # preBoost the data
 (newData, newOut) = preboost(data, outcomes, ratio, iround(newSize*size(data,1)))
 
 # This to get the edgeLogProbs
 (_, edgeLogProbs) =	trainAODE(newData,newOut,LUT) 
 
	#= To Do: Implement using LMDB.jl
 	See http://wildart.github.io/LMDB.jl/manual/
	 =#
 try
	outputFileData = @sprintf("%s_params.txt",sessionId)
	outputFileLUT = @sprintf("%s_LUT.txt",sessionId)
	
	writeArray(outputFileData,edgeLogProbs,Float64)
	writeArray(outputFileLUT,LUT,Int64)
 catch
 	RET_MSG = @sprintf("ERROR: creating files %s*.txt", sessionId)
 finally
 end
 
 return(edgeLogProbs,LUT,RET_MSG)
end
#----------------------------------------------------------------------
# Version of quinticPredict that reads params from file
#---------------------------------------------------------------------
function quinticPredict(data::Array{Int64}, uniqY::Array{Int64}, sessionId::ASCIIString, queryId::ASCIIString) 

#edgeLogProbs::Array{Float64}, LUT::Array{Int64})

 # Read parameters needed for prediction ...
 
 #= To Do: Implement using LMDB.jl
 See http://wildart.github.io/LMDB.jl/manual/
 =#
 
 edgeDataFile = @sprintf("%s_params.txt",sessionId)
 LUTDataFile = @sprintf("%s_LUT.txt",sessionId)
 
  try
  	edgeLogProbs = readArray(edgeDataFile)
  	LUT = readArray(LUTDataFile)
  catch
  	@printf("ERROR: reading files %s*.txt", sessionId)
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
 (_,nodeLogProbs) = getNBparamsAODE(edgeLogProbs,LUT)
 
 # predict in a single shot ...
 (classIndex, predProb, predVal) = predictAODE(cvSet, nodeLogProbs,edgeLogProbs )
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


#----------------------------------------------------------------------
# Version of quinticPredict that reads params as parameters
#---------------------------------------------------------------------
function quinticPredict(data::Array{Int64}, uniqY::Array{Int64}, sessionId::ASCIIString, queryId::ASCIIString,edgeLogProbs::Array{Float64}, LUT::Array{Int64})

 # Read parameters needed for prediction ...
 
 #= To Do: Implement using LMDB.jl
 See http://wildart.github.io/LMDB.jl/manual/
 =#
 
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
 (yLogProbs,nodeLogProbs) = getNBparamsAODE(edgeLogProbs, LUT)
 
 # predict in a single shot ...
 (classIndex, predProb, predVal) = predictAODE(cvSet, nodeLogProbs,edgeLogProbs)
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
 
