
# Version 0.0
# Date: 09-06-15 at 06:54:54 PM IST
#include("Causes.jl")
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
 S(x,y) = (@sprintf("{\"Session ID\": %s, \"Session\": [\n%s]}\n",x,y))
 
 # Query block
 Q(x,y) = (@sprintf("{\"Query ID\": %s, \"Query\": [\n%s]}\n",x,y))
 
 # Candidate block: a= Candidate index in data; b= predClass; c= predProb; d=report
 C(a,b,c,d) = (	 @sprintf("{\"Candidate Index\": \"%s\",\n 
 							\"Predicted Class\": \"%s\",\n
 						\"Predicted Probability\": \"%s\",\n
 						\" Report\": [\n %s]}\n",a,b,c,d))
 
 # Report block:
 R(x,y) = (@sprintf("{\"NODE\": [\n %s]},{\"EDGE\": [\n %s]\n}\n",x,y))
 
 
 
 N(x,y,z) = (@sprintf("{\"column No.\": \"%s\", \"Value\": \"%s\",\n
 						\"Probability\": \"%s\"}\n",x,y,z))
 # Edge block
 E(a,b,c,d,x) = (@sprintf("{\"column No.\": \"%s\", \"Value\": \"%s\",\n
 							\"column No.\": \"%s\", \"Value\": \"%s\",\n
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
 #try
	outputFileData = @sprintf("%s_params.jla",sessionId)
	outputFileLUT = @sprintf("%s_LUT.jla",sessionId)
	
	writeArray(outputFileData,edgeLogProbs,Float64)
	writeArray(outputFileLUT,LUT,Int64)
 #catch
 	#RET_MSG = @sprintf("ERROR: creating files %s*.txt", sessionId)
 #finally
 #end
 
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
 
  #try
  	edgeLogProbs = readArray(edgeDataFile)
  	LUT = readArray(LUTDataFile)
  #catch
  	#@printf("ERROR: reading files %s*.txt", sessionId)
  #end
 
 return quinticPredict(data::Array{Int64}, uniqY::Array{Int64}, sessionId::ASCIIString, queryId::ASCIIString,edgeLogProbs::Array{Float64}, LUT::Array{Int64})
 
 
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
 		getCauses(predClass[selected], cvSet[selected,:], edgeLogProbs, uniqY,LUT)
 end
 
 # Now proceed to generate JSON string ...
 return JSONstr = getPredJSONstr(sessionId, queryId, candData)
end

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------	
	
function getCauses(predClass::Int64, Candidate::Array{Int64}, edgeLogProbs::Array{Float64}, classes::Array{Int64}, LUT::Array{Int64})

 # edgeLogProbs is a numNodes x numNodes x num_class matrix containing log(.) of
 # each combination p(x_1, x_2, y_i) y_i in  set of all possible outcomes, and
 # x_1,x_2 are nodes
 # predClass is a scalar in {0 .. (num_classes-1)}
 # containing the predicted class
 #-----------------------------------------------------------------------
 numNodes = size(edgeLogProbs,1)
 num_class = size(edgeLogProbs,3)
 Candidate = vec(Candidate) # Convert to vector
 num_cols = maximum(LUT[:,3])
 indexPredClass = find(z->z==predClass,classes)[1]
 
 
 (yLogProbs, nodeLogProbs) = getNBparamsAODE(edgeLogProbs,LUT)
 edgeJointProbs = exp(edgeLogProbs)
 # The exponentiation will cause diagonals to be 1.0
 # Need to restore those to 0.0
 for k =1:num_cols
 	colindex = find(z->z==k,LUT[:,3])
 	edgeJointProbs[colindex,colindex,:] = 0.0
 end 
 
 nodeJointProbs = exp(nodeLogProbs)
 yProbs = exp(yLogProbs)
 
 # Conditional probs = Prob(y==[0,1]|node)
 
 nodeCondProb = zeros(Float64,numNodes,num_class)
 nodeCondProb = nodeJointProbs./repmat(sum(nodeJointProbs,2),1,num_class)
 
  
 
 edgeCondProb = zeros(Float64,numNodes,numNodes)
 tempEJP = reshape(edgeJointProbs[:,:,indexPredClass],numNodes,numNodes)
 
 for k=1:(numNodes-1), l =(k+1):numNodes
 	normalizingConst = sum(edgeJointProbs[k,l,:])
 	if (normalizingConst == 0.0)
 		edgeCondProb[k,l] = edgeCondProb[l,k] = 0.0
 	else
 		edgeCondProb[k,l] = edgeCondProb[l,k] =
 		tempEJP[k,l]/normalizingConst
 	end
 end
 
   
 nodeCause = zeros(Float64,num_cols,3)
 edgeCause = zeros(Float64,(div(num_cols*(num_cols-1),2), 5))
 
 
 kmax = maximum(LUT[:,2]) + 1
  
 for k = 1:num_cols
 	nodeVal = Candidate[k]
  	nodeCause[k,1] =	convert(Float64, LUT[nodeVal,3])# Column Number
 	nodeCause[k,2] =	convert(Float64, mod(LUT[nodeVal,1],kmax))  # Value
 	nodeCause[k,3] =	nodeCondProb[nodeVal,indexPredClass]
 end
 
  
 
 count = 1
 for k =1:(num_cols-1),l = (k+1):num_cols
 	valk = Candidate[k]; vall = Candidate[l]
 	edgeCause[count,1] = float64(LUT[vall,3])# Column Number
 	edgeCause[count,2] = float64(mod(LUT[vall,1],kmax)) # Value
 	edgeCause[count,3] = float64(LUT[valk,3])# Column Number
 	edgeCause[count,4] = float64(mod(LUT[valk,1],kmax)) # Value
 	edgeCause[count,5] = edgeCondProb[vall,valk]
 	count += 1
 end
 
 # two scenarios play out ...
 if (predClass == 1)
 # We are interested in the TOP 5 indicators for the problem
 	cutoff = 0.75
 	
 	# Sort descending ...
 	nodeCause = sortrows(nodeCause, by=z->z[3],rev=true)	
 	edgeCause = sortrows(edgeCause, by=z->z[5],rev=true)
 	
 	nidx = findfirst(z->z <= cutoff,nodeCause[:,3])
 	
 	eidx = findfirst(z->z <= cutoff,edgeCause[:,5])
 	
 	nidx = nidx > 5? 5 : nidx -1
 	eidx = eidx > 5? 5 : eidx -1
 
 	return (nidx <= 0?nothing:nodeCause[1:nidx,:], 
 		 eidx <= 0?nothing:edgeCause[1:eidx,:])
 		 
 else # predClass == 0
 	# We are still interested in 5 indicators for the problem
 	# but these will be the BOTTOM 5!
 	cutoff = 0.25
 	
 	# Sort ascending ...
 	nodeCause = sortrows(nodeCause, by=z->z[3])	
 	edgeCause = sortrows(edgeCause, by=z->z[5])
 	 	
 	nidx = findfirst(z->z >= cutoff,nodeCause[:,3])
 	
 	eidx = findfirst(z->z >= cutoff,edgeCause[:,5])
 	
 	
 	nidx = nidx > 5? 5 : nidx -1
 	eidx = eidx > 5? 5 : eidx -1
 	
 	if (nidx > 0)
 		nodeCause[1:nidx,3] = 1.0 .- nodeCause[1:nidx,3]
 	end
 	
 	if (eidx > 0)
 		edgeCause[1:eidx,5] = 1.0 .- edgeCause[1:eidx,5]
 	end
 	 	
 	return (nidx <= 0?nothing:nodeCause[1:nidx,:], 
 		 eidx <= 0?nothing:edgeCause[1:eidx,:])
 end 		 
 		 
 		 
end

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------	
function heatAnalysis(edgeLogProbs::Array{Float64}, LUT::Array{Int64})

 numNodes = size(edgeLogProbs,1)
 num_class = size(edgeLogProbs,3)
 numCols = maximum(LUT[:,3])
 
 (_, nodeLogProbs) = getNBparamsAODE(edgeLogProbs,LUT)
 edgeJointProbs = exp(edgeLogProbs)
 
 
 nodeJointProbs = exp(nodeLogProbs)
 
 
 # Conditional probs = Prob(y==[0,1]|node)
 
 nodeCondProb = zeros(Float64,numNodes)
 nodeProbs = sum(nodeJointProbs,2)
 nodeCondProb = vec(nodeJointProbs[:,2]./repmat(nodeProbs,1,num_class))
 nodeProbs = vec(nodeProbs)
 
 edgeCondProb = zeros(Float64,numNodes,numNodes,num_class)
 
 SEJP = reshape(sum(edgeJointProbs,3),numNodes,numNodes)
 for k =1:num_class
 	ECP = reshape(edgeJointProbs[:,:,k],numNodes,numNodes)./SEJP
 	edgeCondProb[:,:,k] = reshape(ECP,numNodes,numNodes,1)
 end
 	
 	
 #edgeCondProb = edgeJointProbs./repmat(sum(edgeJointProbs,3),1,1,num_class)
 
 nodeHullIndex = zeros(Float64,numNodes)
 nodeHullIndex = log(nodeCondProb./(1.0 .- nodeCondProb ))
 
 
 edgeHullIndex = zeros(Float64,numNodes,numNodes)
 tempEdgeCondProb = reshape(edgeCondProb[:,:,2],numNodes,numNodes)
 edgeHullIndex = abs(log(tempEdgeCondProb./(1.0 .- tempEdgeCondProb)))
 println(maximum(edgeHullIndex))
 
 colHeat = zeros(Float64,numCols,2)
 
 for k=1:numCols
 	colindex = find(z->z==k,LUT[:,3])
 	# find the node with the maximum heat in this
 	 (_,idx) = findmax(abs(nodeHullIndex[colindex]))
 	 colHeat[k,1] = nodeHullIndex[colindex[idx]]
 	 colHeat[k,2]  = sum(nodeProbs[colindex].*abs(nodeHullIndex[colindex]))
 end
 
 
 edgeNodeList = Int64[]
 threshold = 5.0
 for k =1:(numNodes-1), l =(k+1):numNodes
   	 if (edgeHullIndex[k,l] > threshold)
   	 	push!(edgeNodeList,k)
   	 	push!(edgeNodeList,k)
   	 end
 end
 columnList = sort(unique(LUT[edgeNodeList,3]))
 return colHeat, columnList
end


 
