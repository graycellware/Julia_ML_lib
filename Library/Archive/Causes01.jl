function getCauses(edgeLogProbs::Array{Float64}, predClass::Int64, Candidate::Array{Int64}, LUT::Array{Int64})
# Version 1.0
# Date: 13-05-15 at 02:45:36 AM IST

 # edgeLogProbs is a numNodes x numNodes x num_class matrix containing log(.) of
 # each combination p(x_1, x_2, y_i) y_i in  set of all possible outcomes, and
 # x_1,x_2 are nodes
 # predClass is a scalar in {0 .. (num_classes-1)}
 # containing the predicted class
 
 num_cols = length(Candidate)
 edgeJointProbs = exp(edgeLogProbs)
 
 #-----------------------------------------------------------------------
 numNodes = size(edgeLogProbs,1)
 num_class = size(edgeLogProbs,3)
 Candidate = vec(Candidate) # Convert to vector
 
 nodeClassProbs = zeros(Float64,numNodes,num_class)
 nodeProbs = zeros(Float64,numNodes)
 yProbs = zeros(Float64,num_class)
 
 predClassIndex = predClass + 1
 
 # Marginalize to get the node Class Probabilities
 nodeClassProbs = reshape(sum(edgeJointProbs,1),numNodes,num_class)
 nodeProbs = sum(nodeClassProbs,2)

 
 # Normalize nodeClassProbs
 # nodeClassProbs contains P[x_i,y == j]. It is a numNodes x num_class array
 nodeClassProbs ./= repmat(nodeProbs,1,num_class)

 
 
 
 # Marginalize again to get the relative outcome Probabilities
 yProbs = sum(nodeClassProbs,1)
 yProbs ./= sum(yProbs)
 yProbs = vec(yProbs)
   
 
 
 nodeCondProb = zeros(Float64,numNodes)
 edgeCondProb = zeros(Float64, numNodes,numNodes)
 
 # Compute the conditional probabilities
 # we want to identify causes for concern.
 
 nodeCondProb = vec(nodeClassProbs[:,predClassIndex])./nodeProbs
 
 # Nodes are independent. Hence P(x,y)= P(x).P(y)
 denom = repmat(nodeProbs,1,numNodes)
 denom = denom.*denom.'
 
 for k = 1:numNodes
 	denom[k,k] = 1
 end
 #writedlm("Denom.csv",denom,',')
 
 
 
 #=
 	edgeCondProb = P(y=1|edge) = P(y=1 and edge)/P(edge).
 	But edgeCondProb(y=1,edge) + edgeCondProb(y=0,edge) = Prob(edge).
 	Hence Prob(edge) is calculated as:
 =#
 probEdge = sum(edgeJointProbs,3)
 
 # We set the diagonal entries to 1
 for k =1:numNodes
 	probEdge[k,k] = 1
 end
 
 
 edgeCondProb = reshape(edgeJointProbs[:,:,predClassIndex]./probEdge,numNodes,numNodes)
 
 # We reset the diagonal entries to 0 after the division
 for k =1:numNodes
 	edgeCondProb[k,k] = 0
 end
 
 # Prepare output data structures
 
 
 nodeCause = zeros(Float64,num_cols,3)
 edgeCause = zeros(Float64,(num_cols*num_cols), 5)
 
 kmax = maximum(LUT[:,2]) + 1
 value = zeros(Int64,num_cols)
 for k = 1:num_cols
 	value[k] = Candidate[k]
  	nodeCause[k,1] =	convert(Float64, LUT[value[k],3])# Column Number
 	nodeCause[k,2] =	convert(Float64, mod(LUT[value[k],1],kmax))  # Value
 	nodeCause[k,3] =	nodeCondProb[value[k]]
 end
 
  
 
 count = 1
 for k =1:(num_cols-1),l = k:num_cols
 	edgeCause[count,1] = convert(Float64, LUT[value[l],3])# Column Number
 	edgeCause[count,2] = convert(Float64,mod(LUT[value[l],1],kmax)) # Value
 	edgeCause[count,3] = convert(Float64, LUT[value[k],3])# Column Number
 	edgeCause[count,4] = convert(Float64, mod(LUT[value[k],1],kmax)) # Value
 	edgeCause[count,5] = edgeCondProb[l,k]
 	count += 1
 end
 
 
 
 
 
 # two sceanrios play out ...
 if (predClass == 1)
 # We are interested in the TOP 5 indicators for the problem
 	cutoff = 0.95
 	
 	# Sort descending ...
 	nodeCause = sortrows(nodeCause, by=z->z[3],rev=true)	
 	edgeCause = sortrows(edgeCause, by=z->z[5],rev=true)
 
 	nidx = findfirst(z->z <= cutoff,nodeCause[:,3])
 	eidx = findfirst(z->z <= cutoff,edgeCause[:,5])
 	
 	if nidx > 5
 		nidx = 5
 	end
 
 	if eidx > 5
 		eidx = 5
 	end
 
 	return (isempty(nidx)?nothing:nodeCause[1:nidx,:], 
 		 isempty(eidx)?nothing:edgeCause[1:eidx,:])
 		 
 else # predClass == 0
 	# We are still interested in the top 5 indicators for the problem
 	# but these will be the BOTTOM 5!
 	cutoff = 0.95
 	
 	# Sort ascending ...
 	nodeCause = sortrows(nodeCause, by=z->z[3])	
 	edgeCause = sortrows(edgeCause, by=z->z[5])
 	
 	
 	# Ok. We may have the diagonals in, and these will be zeros. So, we start
 	# from non-zero values.
 		
 	
 	znidx = findfirst(z->z > 0.05,nodeCause[:,3])
 	zeidx = findfirst(z->z > 0.05 ,edgeCause[:,5])
 	
 	nidx = findfirst(z->z >= cutoff,nodeCause[:,3])
 	eidx = findfirst(z->z >= cutoff,edgeCause[:,5])
 	
 		
 	nodeCause[znidx:nidx,3] = 1.0 .- nodeCause[znidx:nidx,3]
 	edgeCause[zeidx:eidx,5] = 1.0 .- edgeCause[zeidx:eidx,5]
 	
 	if (nidx - znidx) > 5
 		nidx = znidx + 5
 	end
 
 	if (eidx - zeidx) > 5
 		eidx = zeidx + 5
 	end
 	
 	
 	
 	return (nidx == 0?nothing:nodeCause[znidx:nidx,:], 
 		 eidx == 0?nothing:edgeCause[zeidx:eidx,:])
 end 		 
 		 
 		 
end
 
	 
	 

