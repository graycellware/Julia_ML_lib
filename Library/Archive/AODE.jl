function  trainAODE(X::Array{Int64},y::Array{Int64},LUT::Array{Int64})

# Returns: (nodeLogProbs, edgeLogProbs )

# [nodeLogProbs, edgeLogProbs] = aodeTrain(X,y,metaData)
#	X is assumed encoded
# Implements AODE: Averaged One-Dependence Estimators
# Constraints: Only categorical data in training and test sets
# Cannot be used for large data sets
# Columns take sequential integer values between 0 and k(i) (see below)
# Inputs:
# 	1. X: m x n matrix with m training vectors as rows.
# 	2. y: Outcome vector m x 1.
#	3. LUT: The lookup table for coding/decoding the values of x
#		It is a num_nodes x 3 matrix
#		Column 1: ith entry gives corresponding code: 
#				col = code (div) kmax. value = code (mod) kmax.
#				See Row 2 for kmax.
#		Column 2: ith entry gives the maximum value the corresponding column can take.
#				kmax = 1 + maximum(LUT[:,1])
#		Column 3: ith entry gives which column to which the corresponding node belongs
#		Column 4: the ith entry gives the value by which the corresponding column
#				in the data has to be subtracted
# Outputs:
# 	1. nodeLogProbs: num_classess x numNodes matrix containing log(Prob(x,y)) 
# 	where, num_classess is the number of values the outcome vector y can take,
#	numNodes = Total number of nodes. One for each potential value a column could take.
#	x is a node
# 	2. edgeLogProbs: is a 3D matrix: 
#		num_classess x numNodes x numNodes matrix containing log(Prob(x1,x2, y))
#		x1 and x2 are nodes, not from the same column


	
	
	(num_rows, num_cols) = size(X)
	(lenY,ny) = size(y)
		
	if (num_rows != lenY)
		error("Outcomes do not match in size data")
	end

	if (ny != 1)
		error("Outcome should be a column vector")
	end

	
		
	# Total number of nodes. One for each potential value a column could take.
	numNodes = size(LUT,1)
	##------------------------------------------------------------------------------
	# Find out the setof values that y can take and create a lookup table
	##------------------------------------------------------------------------------
	uniqy = sort(unique(y))
	
	num_classes = length(uniqy) # Number of outcome categories (k)
		
	 
	
	# Create two matrices to store the relative counts
	nodeLogProbs = zeros(Float64, numNodes, num_classes) # 2D matrix for nodes
	edgeLogProbs = zeros(Float64, numNodes,numNodes, num_classes) # 3D matrix for edges
	
	denomN = zeros(Float64,size(nodeLogProbs))
	denomE = zeros(Float64,size(edgeLogProbs))
	
	# Create common data for denominator
	# First for nodeLogProbs
	L = LUT[:,2].*num_classes
	denomN = repmat((L .+ num_rows),1,num_classes)
	
	# Denominator for edgeLogProbs
	L = (LUT[:,2]*L.') .+ num_rows
	for yclass = 1:num_classes
		denomE[:,:,yclass] = reshape(L,numNodes,numNodes,1)
	end
	
	
	# Count the entries
	for yclass = 1:num_classes
		yidx = find(z->z == uniqy[yclass],y)
		Xy = X[yidx,:]
		numRows = length(yidx)
		
		for row = 1:numRows
			colData = vec(Xy[row,:])
			nodeLogProbs[colData,yclass] += 1.0
			edgeLogProbs[colData,colData,yclass] += 1.0
		end
	end
	
	nodeLogProbs = log((nodeLogProbs+1.0)./denomN)
	edgeLogProbs = log((edgeLogProbs+1.0)./denomE)
	
	#----------------------------------------------
	## Apply Mask
	#----------------------------------------------
	# There are no edges between intrabank nodes
	# And edges are formed only across nodes within 
	# the same row of training data. 
	# Let us reset those entries to 0 in edgeLogProbs
	#----------------------------------------------
	
	for col = 1:num_cols
		cidx = find(z->z == col, LUT[:,3])
		edgeLogProbs[cidx,cidx,:] = 0 
	end
	##----------------------------------------------------------------------------
	
	return (nodeLogProbs, edgeLogProbs)
end

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
function 	predictAODE(X::Array{Int64},
						nodeLogProbs::Array{Float64}, 
						edgeLogProbs::Array{Float64})

# aodePredict(X, nodeLogProbs, edgeLogProbs, LUT)
# Returns: ( predClassIndex, predProb, probValues) = 
# Implements prediction in AODE
# 	Constraints: Only categorical data in training and test sets
# 	Cannot be used for large data sets
# 	Columns take sequential integer values between 0 and k(i) (see below)
# Inputs:
# 	1. X: num_rows x num_cols matrix 
#	with num_rows candidate vectors for prediction as rows.
#	2. nodeLogProbs: numNodes x num_classes matrix containing log(Prob(x,y)); 
# 	where, num_classes is the number of values the outcome vector y can take
#	numNodes = Total number of nodes. One for each potential value a column could take.
#	x is a node
# 	3. edgeLogProbs: is a 3D matrix: numNodes x numNodes x num_classes matrix
#	 containing log(Prob(x1,x2, y)); x1 and x2 are nodes, not from the same column
#	
# Outputs:
#	1. predClassIndex: This is a num_rows entries vector which gives a value 
#		between 1 to maxYValue as index into a SORTED array of possible values 
#		(outcome classes). The ith entry is the index into this table 
#		for the ith row of X.
#	2. predProb: a num_rows entries vector. The ith entry is the 
#			predicted probability that the ith row of X
#			belongs to the class given in the ith row of predClassIndex. 
#			predProb = max(probvalues[row,:]);
#	3. probValues: num_rows x num_classes matrix, whose ith row and jth column 
#			gives the probability that row i of X belongs to the class j. 
#			1 <= j <= maxYValue. j is an index into a SORTED array 
#			of possible values (outcome classes)


	(num_rows,num_cols) = size(X)
	
		
	# derive important parameters
	 num_classes = size(edgeLogProbs,3)
	 numNodes =  size(edgeLogProbs,2)
	 
	 
	 	
	# Initialize output variables
	predClassIndex = zeros(Int64,num_rows)
	predProb = zeros(Float64,num_rows)
	probValues = zeros(Float64, num_rows, num_classes)
	
	for row = 1:num_rows
		 colData = vec(X[row,:])
		 edgeSum = reshape(sum(edgeLogProbs[colData,colData,:],1), 
		 				length(colData), num_classes)
		 
		 
		 sumEdgeProbs = sum(exp(nodeLogProbs[colData,:].*(1-num_cols) + edgeSum),1)
		 		 
		 probValues[row,:] = sumEdgeProbs./sum(sumEdgeProbs)
		 
		 (predProb[row], predClassIndex[row]) = 
										findmax(probValues[row,:])
	end
	
	return (predClassIndex, predProb, probValues)
		
end

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------			

function getNBparamsAODE(	nodeLogProbs::Array{Float64}, 
							edgeLogProbs::Array{Float64})			
	num_classes = size(nodeLogProbs,2)
	yLogProbs = zeros(Float64, 1, num_classes)
	yLogProbs = sum(nodeLogProbs,1)
	yLogProbs ./= sum(yLogProbs)
	return (yLogProbs, nodeLogProbs)
end				
				
	
	
	
	 
	 

