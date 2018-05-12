#!/usr/bin/julia
# Version 4.0
# Date: 10-05-15 at 07:46:04 PM IST


#---------------------------------------------------------------------------
function makemetadata(X::Array{Int64})
#		metaData : n x 3 matrix 
#		Column 1: ith entry gives minimum value that column i of X can take (assume all 0)
# 		Column 2: ith entry gives maximum value that column i of X can take
#		Column 3: ith entry gives number of values that column i of X can take
#			We assume rows take values between 0 and k(i)-1 
#			where k(i) is the entry in column 2, row i of metaData.
	n = size(X,2)	# Number of columns of X
	metaData = zeros(Int64,n,3)
	metaData[:,2] = (maximum(X,1) - minimum(X,1)).'
	metaData[:,3] = (metaData[:,2] .+ 1).'
	return metaData
end
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function encodeX!(X::Array{Int64},metaData::Array{Int64})

# encodeX!(X::Array{Int64},metaData::Array{Int64})
#	MODIFIES X
# 	Encodes the values of X (assumed only categorical)
# Inputs:
# 	X: m x n matrix with m training vectors as rows.
# 	
# 	metaData : n x 3 matrix 
#		Column 1: ith entry gives minimum value that column i of X can take (assume all 0)
# 		Column 2: ith entry gives maximum value that column i of X can take
#		Column 3: ith entry gives number of values that column i of X can take
#			We assume rows take values between 0 and k(i)-1 
#			where k(i) is the entry in column 2, row i of metaData.
# Outputs:
#	1. Encoded X
#	
#	2. LUT: The lookup table for coding/decoding the values of x
#		It is a num_nodes x 3 matrix
#		Column 1: ith entry gives corresponding code: 
#				col = code (div) kmax. value = code (mod) kmax.
#				See Row 2 for kmax.
#		Column 2: ith entry gives the maximum value the corresponding column can take.
#				kmax = 1 + maximum(LUT[:,1])
#		Column 3: ith entry gives which column to which the corresponding node belongs


	(num_rows,num_cols) = size(X)
	kmax = maximum(metaData[:,2]) + 1     # largest value any column can take is (kmax -1)
	num_nodes = sum(metaData[:,2]) + num_cols 	  # Total number of nodes. One for each potential value a column in X could take
	
	##------------------------------------------------------------------------------
	#	Create LUT
	##------------------------------------------------------------------------------
	# Create a lookup table for index:-> code, max values, col number
	
	LUT = zeros(Int64, num_nodes,4)
	
	rowIndx = 1
	for column_in_X = 1:num_cols
		k = metaData[column_in_X,3]		# k is the number of values that the 
										# corresponding column of X can take.
		LUT[rowIndx:(rowIndx+k-1), 1] = [0:(k-1)] .+ (column_in_X*kmax) # Coded value
		LUT[rowIndx:(rowIndx+k-1), 2] = ones(Int64,k).*k 	# store the max values per column
		LUT[rowIndx:(rowIndx+k-1), 3] = ones(Int64,k).*column_in_X 	# store the column numbers
		rowIndx += k
	end
	
	# Ensure that the columns of X range between 0 and (k-1)
	X_Col_Min = minimum(X,1)
	for k = 1:num_nodes
		col_id = LUT[k,3]
		LUT[k,4] = X_Col_Min[col_id]
	end
	
	X -= repmat(X_Col_Min, num_rows,1)
	
	##----------------------------------------------------------------------------
	# Encode input data
	# Genrate column base number = column number x kmax
	##----------------------------------------------------------------------------
	colNumbers = [1:num_cols].*kmax
	X += repmat(colNumbers.',num_rows,1) 	# All the input data has now been encoded
									# Remember data in X is along rows
									# Features are in columns
									
	# Re-encode X to contain the LUT index numbers
	for idx = 1:num_nodes
		X[find(z->z == LUT[idx,1], X)] = idx
	end
	# Now the entries in X are the corresponding indices in LUT[1,:]
	 return X, LUT
end

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function encodeX(X::Array{Int64},metaData::Array{Int64})

# encodeX(X::Array{Int64},metaData::Array{Int64})
# DOES NOT MODIFY X. RETURNS A ENCODED VERSION OF X
# 	Encodes the values of X (assumed only categorical)
# Inputs:
# 	X: m x n matrix with m training vectors as rows.
# 	
# 	metaData : n x 3 matrix 
#		Column 1: ith entry gives minimum value that column i of X can take (assume all 0)
# 		Column 2: ith entry gives maximum value that column i of X can take
#		Column 3: ith entry gives number of values that column i of X can take
#			We assume rows take values between 0 and k(i)-1 
#			where k(i) is the entry in column 2, row i of metaData.
# Outputs:
#	1. Encoded X
#	
#	2. LUT: The lookup table for coding/decoding the values of x
#		It is a num_nodes x 3 matrix
#		Column 1: ith entry gives corresponding code: 
#				col = code (div) kmax. value = code (mod) kmax.
#				See Row 2 for kmax.
#		Column 2: ith entry gives the maximum value the corresponding column can take.
#				kmax = 1 + maximum(LUT[:,1])
#		Column 3: ith entry gives which column to which the corresponding node belongs
#		Column 4: the ith entry gives the value by which the corresponding column
#				in the data has to be subtracted

	Z = zeros(Int64, size(X))
	Z = 0 .+ X
	return encodeX!(Z, metaData)
end

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function trainNB(X::Array{Int64},y::Array{Int64}, LUT::Array{Int64})

	(num_rows, num_cols) = size(X)
	num_nodes = size(LUT,1)
	
	target_classes = sort(unique(y))
	num_classes = length(target_classes)
	##------------------------------------------------------------------------------
	# Define the other output matrices
	##------------------------------------------------------------------------------
	yLogProbs = zeros(Float64, 1,num_classes) # Row vector
	nodeLogProbs = zeros(Float64, num_nodes, num_classes) # 2-D matrix
	ycount = zeros(Float64,1,num_classes)
	
	##------------------------------------------------------------------------------
	# Populate nodeLogProbs
	##------------------------------------------------------------------------------
	for y_class = 1:num_classes
		yidx = find(z->z == target_classes[y_class], y)
		ycount[1,y_class] = convert(Float64,length(yidx))
		
		#--------------------------------------------
		# Compute yLogProbs (with Lapalace estimator)
		#--------------------------------------------
		yLogProbs[1, y_class] = log((length(yidx)+1.0)/(num_rows+num_classes))
		
		#--------------------------------------------
		# Calculate count of node given y_class
		#--------------------------------------------
				
		for node_value = 1:num_nodes
			# Get column to search
			col = LUT[node_value,3]
			nodeLogProbs[node_value, y_class] += 
						convert(Float64,length(find(z->z == node_value, X[yidx,col] )))
		end
	end
	#
	#------------------------------------------------------------------------------
	# Apply Lapalce estimators to nodeLogProbs
	# numerator is easy -- just add 1
	# denominator is K + num_classes*(number of values column can take)
	# The number of values the column can take is already factored in LUT(2,:)
	##------------------------------------------------------------------------------
	#denom = repmat(LUT[:,2].*num_classes,1,num_classes) .+ repmat(ycount,num_nodes,1)
	denom = repmat(LUT[:,2].*num_classes,1,num_classes) .+ num_rows
	nodeLogProbs = log((nodeLogProbs+1.0)./denom)
	##------------------------------------------------------------------------------
	
	return (yLogProbs, nodeLogProbs)
end

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function predictNB(X::Array{Int64}, yLogProbs::Array{Float64}, 
					nodeLogProbs::Array{Float64})

# Implements prediction in Naive Bayes
# Constraints: 
#	1. Only categorical data in training and test sets
# 	2. Cannot be used for large data sets
# Columns take sequential integer values between 0 and k(i) (see below)
# Inputs:
# 	1. X: mx x nx matrix with mx candidate vectors for prediction as rows.
#	2. yLogProbs: maxYValue x 1 column vector. 
#		The ith entry is the log(prob(y==(i-1))).
#	3. nodeLogProbs: maxYValue x numNodes matrix. The entry [i,j] gives the log of the
#		joint probability that y == (i-1) and x == unencoded value of j

# Outputs:
#	1. predClassIndex: This is a mx x 1 column vector 
#		which gives a value between 1 to maxYValue as
#		index into a SORTED array of possible values (outcome classes). 
#		The ith entry is the index into this table for the ith row of X.
#	2. predProb: mx x 1 column vector. 
#		The ith entry is the predicted probability that the ith row of X
#		belongs to the class given in the ith row of predClassIndex. 
#		predProb = max(probvalues[row,:])
#	3. probValues: mx x maxYValue matrix, 
#			whose ith row and jth column gives the ptobability that row i of X
#			belongs to the class j. 1 <= j <= maxYValue. 
#			j is an index into a SORTED array of possible values (outcome classes)


	(num_candidates, num_features) = size(X)
	
	
	# derive important parameters
	( numNodes, maxYValue ) = size(nodeLogProbs)
	
	
	# Initialize output variables
	predClassIndex = zeros(Int64, num_candidates)
	predProb = zeros(Float64, num_candidates)
	probValues = zeros(Float64, num_candidates, maxYValue)
	
	for candidate = 1:num_candidates
	
		candData = vec(X[candidate,:])
		sumProbs = exp((1-num_features).*yLogProbs + sum(nodeLogProbs[candData,:],1))
		probValues[candidate,:] = sumProbs./sum(sumProbs) # Normalizing across y
		(predProb[candidate], predClassIndex[candidate]) = 
										findmax(probValues[candidate,:])
	end
	
	return (predClassIndex, predProb, probValues)
end
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
		
function encodeCandMatrix!(X::Array{Int64}, LUT::Array{Int64})

#=
 The training sample X has been encoded.
 The test sample therefore has to be similarly encoded using the same
 parameters used to encode X.
	
 The input matrix is the test matrix.
 The parameters are stored in LUT.
 
 The ouput is the encoded test matrix, ready for classification.
=#
 num_cols = maximum(LUT[:,3])
 if (size(X,2) != num_cols)
 	error("Incompatible column size")
 end
	
 min_col_value = zeros(Int64,num_cols)
	
 for k = 1:num_cols
 	idx = findfirst(z->z==k,LUT[:,3])
	min_col_value[k] = LUT[idx,4]
 end
	
	
 num_rows = size(X,1)
 X -= repmat(min_col_value.',num_rows,1)
 kmax = maximum(LUT[:,2])
	
 # First stage of encoding
 colNumbers = [1:num_cols].*kmax
 X += repmat(colNumbers.',num_rows,1) 
	
 # Re-encode X to contain the LUT index numbers
 for idx = 1:num_nodes
	X[find(z->z == LUT[idx,1], X)] = idx
 end
 # Now the entries in X are the corresponding indices in LUT[1,:]

end
		
	
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------	

function getCVdata(sampleFraction::Float64,X::Array{Int64},y::Array{Int64})
#	gentestset(sampleFraction,X,y)
#	Returns: (trainSet, trainOutcome, testSet, testOutcome)
# Takes as input the desired percentage of the training set to be used for
# 		cross validation.  Sampling without replacement. 
# 		0 < sampleFraction < 1. Typically between 0.1 (10%) to 0.15 (15%)
# The input training set (X) is a m x n matrix
# The outcome set (y) is a m x 1 binary matrix
# Outputs: [ trainSet, trainOutcome, testSet, testOutcome ] 
#	trainSet: Training set
#	trainOutcome: Corresponding outcome vector derived from y
#	testSet: The Cross validation set 
#	testOutcome: The outcome vector for testSet
  
 if ((sampleFraction > 0.5) || (sampleFraction <= 0.0))
 	error("Invalid fraction for cross validation set")
 end
 
 (num_rows, num_cols) = size(X)
 sampleSize = iround(num_rows*sampleFraction)
  
 testSet = zeros(Int64, sampleSize, num_cols)
 testOutcome = zeros(Int64, sampleSize,1)
 trainSet = zeros(Int64, (num_rows - sampleSize), num_cols)
 trainOutcome = zeros(Int64, (num_rows - sampleSize), 1)
  
  
 # sample sampleSize numbers without replacement
 Idx = randperm(num_rows)
 testIdx = Idx[1:sampleSize]
  
 # Based on the sample indices create test set and the corresponding test outcome set
 testSet += X[testIdx,:];
 testOutcome += y[testIdx,:];
 
 whole = [1:num_rows]
 restIdx = setdiff(whole,testIdx)
 trainSet += X[restIdx,:]
 trainOutcome += y[restIdx,:]
  
 return (trainSet, trainOutcome, testSet, testOutcome)
  
end


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function performanceStats(expectedClass::Array{Int64}, observedClass::Array{Int64})
# Returns (TPR, TNR, Accuracy)

 EL = length(expectedClass)
 OL = length(observedClass)
 if (EL != OL)
 	error("Inputs are not of same size")
 end
 
 
 hits = zeros(Float64,OL)
 hits = [(observedClass[k]*expectedClass[k]) == 1?1.0:0.0 for k = 1:OL]
 truePos = sum(expectedClass)
 TP = sum(hits)
 TPR = TP/truePos

 
 hits = zeros(Float64,OL)
 hits = [((1-observedClass[k])*(1-expectedClass[k])) == 1?1.0:0.0 for k = 1:OL]
 trueNeg = OL -TP
 TN = sum(hits)
 TNR = TN/trueNeg
 
 
 
 Accuracy = 0.5*(TP+TN)/convert(Float64,EL)
 return (TPR, TNR, Accuracy)
end
 
 
 
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function getThreshold(X::Array{Int64}, y::Array{Int64}, LUT::Array{Int64})
#	Returns the threshold that maximizes sqrt(TPR * TNR)
 uniqY = sort(unique(y))

 maxIters = 100
 step = 0.01
 threshValues = zeros(Float64,maxIters)
 range = [0.0:step:1.0]
 lenR = length(range)
 frac =0.1
 num_data = size(X,1)
 lenP = iround(num_data * frac)


 for iter = 1:maxIters
	(trnSet, trnOut, cvSet, cvOut )	= getCVdata(frac,X,y)
	(yLogProbT, nodeLogProbT) =	trainNB(trnSet,trnOut,LUT)
	(predClassIndex, predProb, probValues) = predictNB(cvSet,yLogProbT, nodeLogProbT)
	# ----------- actual thresholding process --------------------------------------
	# We ignore the predClassIndex totally
	#-------------------------------------------------------------------------------
		
	cThresh = zeros(Float64,lenR)
	for index =1:lenR
		z = range[index]
		E = predProb .> z
		Cmp = [E[k]?1:0 for k =1:lenP]
		hits = zeros(Float64,lenP)
		hits = [Cmp[k]*cvOut[k] == 1?1.0:0.0 for k = 1:lenP]
		trueHits = convert(Float64,sum(cvOut,1)[1,1])
		TP = sum(hits)/trueHits
		
		
		hits = zeros(Float64,lenP)
		hits = [(1-Cmp[k])*(1-cvOut[k]) == 1?1.0:0.0 for k = 1:lenP]
		trueHits = convert(Float64,sum((1 .-cvOut),1)[1,1])
		TN = sum(hits)/trueHits
		
		cThresh[index] = sqrt(TP*TN)
	end
	(_,index) = findmax(cThresh)
	threshValues[iter] = range[index]
 end
 return (mean(threshValues), std(threshValues))
end


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
function getNewProb(oldProbs::Array{Float64}, threshold::Float64)

# getNewProb(oldProbs, threshold)
#	ASSUMES BINARY
# Returns: (newClassIndex, newProbs)
# Recomputes probability based on threshold
# Inputs:
#	1. oldProbs: m column vector giving the probabilities of determined class
#	2. threshold: scalar value indicating threshold. 0 < threshold < 1
# Outputs:
#	1. newClass: Output class recomputed as newProbs > 0.5. newClass lies in {0,1}
#	2. newProbs: m x  column vector whose ith entry is the recomputed value
#			for the ith entry of the input oldProbs.

	# Model newProbs = 1-exp(-alpha*oldProbs)/(1-exp(-alpha))
	#	when oldProbs == threshold, newProbs == 0.5
	# We determine alpha through a quick search of a the Lookup table
	
 
 
 if ((threshold < 0.0) || (threshold > 1.0))
 	error("0 < threshold < 1 ")
 end
 
 Cmp = oldProbs .> threshold
 newClass = [ Cmp[z]?1:0 for z =1:length(oldProbs)]
 newClassIndex = newClass .+ 1 # Assumes we will get class fom a sorted array
 
 if ( threshold == 0.5)
 	return (newClassIndex, oldProbs)
 end
 
 newProbs = zeros(Float64,size(oldProbs))
 
 
 if ( threshold < 0.0001)
 	newProbs += oldProbs
 	nidx = find(z->z==1,Cmp)
 	newProbs[nidx] = 1.0
	return (newClassIndex, newProbs)
 end
 
 if( threshold > 0.99999)
	newProbs += oldProbs
 	nidx = find(z->z < threshold,Cmp)
 	newProbs[nidx] =0.0
	return (newClassIndex, newProbs)
 end
 #-------------------------------------------
 # Compute alpha
  #-------------------------------------------	
 	# Localize threshold using lookup table
 	LUT = [0.0001 6931.5;
	0.050000   13.863000;
    0.100000   6.922000;
    0.150000    4.551000;
    0.200000    3.281000;
    0.250000    2.438000;
    0.300000    1.801000;
    0.350000    1.279000;
    0.400000    0.822000;
    0.450000    0.403000;
    0.500000    0.500000;
    0.550000   -0.403000;
    0.600000   -0.822000;
    0.650000   -1.279000;
    0.700000   -1.801000;
    0.750000   -2.438000;
    0.800000   -3.281000;
    0.850000   -4.551000;
    0.900000   -6.922000;
    0.950000  -13.863000;
    0.99999   -709.78]
 
 
 f(x) = (abs((1-exp(-threshold*x))./(1-exp(-x)) -0.5))
 
 idx = findfirst(z->z >= threshold, LUT[:,1])
 
 if(threshold == LUT[idx,1])
	 alpha = LUT[idx,2]
 else 
 	Ulim = LUT[(idx-1),2] 	# Upper bound for alpha
 	Llim = LUT[idx,2]		# Lower bound for alpha
 	step = 0.0001 # Accuracy to 4th place of decimal
 	R = Llim:step:Ulim
 	
 	minVal = 1000000
 	for k in R
 		comp = f(k)
 		if (comp < minVal)
 			alpha = k
 			minVal = comp
 		end
 	end
 end
 
 #------------------------------------------------------
 # We have alpha
 # Compute newProbs from oldProbs using model
 #------------------------------------------------------
 denom = 1- exp(-alpha)
 newProbs = (1 - exp(-alpha.*oldProbs))./denom
 return (newClassIndex, newProbs)
end


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

function predictNB(X::Array{Int64},yLogProb::Array{Float64}, 
						nodeLogProb::Array{Float64}, threshold::Float64)
# Naive Bayes predictor with threshold
#	ASSUMES BINARY
# Takes as input:
#= 
	X::Array{Int64}: The input set to be predicted (encoded)
	yLogProb::Array{Float64}: The log Probabilities of the 2-classes
	nodeLogProb::Array{Float64}: The log Probabilities of the nodes
	threshold::Float64 : scalar threshold value

	Returns: (newClassIndex, newProbs, ProbVals)
=#

 (_, predProb, _) = predictNB(X,yLogProb, nodeLogProb)
 (newClassIndex, newProbs) = getNewProb(predProb, threshold)

 ProbVals = zeros(Float64,length(predProb),2)
 for k = 1:length(predProb)
 	ProbVals[k, newClassIndex[k]] = newProbs[k]
 	ProbVals[k,(3 - newClassIndex[k])] = 1.0 - newProbs[k]
 end
 
 return (newClassIndex, newProbs, ProbVals)
end
#---------------------------------------------------------------------------
				
	
	
	
	 
	 



