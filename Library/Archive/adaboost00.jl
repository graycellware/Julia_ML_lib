function  getNewWeights(yp::Array{Int64},ya::Array{Int64}, weights::Array{Float64})
# Version 0.0
# Date: 09-06-15 at 06:54:13 PM IST

# function W = getNewWeights(yp,ya)
# Returns: (alpha, W)
# Assigns a weight to each data element based on whether the classifier has 
# classifier has predicted it accurately or not.
# Inputs:
#	yp is a num_data x 1 column vector. It is the predicted class of the classifier.
#	ya is a num_data x 1 column vector. It is the actual class.
# 	yp, ya in {0,1}
#	W is a num_data x 1 column vector. W(i) is the old weight for the row i in the base data set.
#	idx is the indices of the items from the base data set that were selected for training
# Outputs:
#	W is a num_data x 1 column vector. W(i) is the new weight for the row i in the base data set.
#

FLIP_FLAG = false


W = vec(weights./sum(weights))
num_data = length(yp) # Number of outcomes = number of data elements in base set

if length(ya) != num_data
	error("Mismatch in lengths of predicted and actual outcome vectors")
end

# Compute alpha = 0.5*ln((1-epsilon)/epsilon) epsilon is the error rate


E = [ yp[k] != ya[k]?1.0:0.0 for k = 1:num_data] # E = 0 if yp == ya, E = 1 if yp != ya.

epsilon = 0.0
# How does this work:
# We always use the Base Training Set to compute the error.
# So error is proportional to the weight of each data element in the BTS.
for k = 1:num_data
	if (yp[k] != ya[k])
		epsilon += W[k]
	end
end


if (epsilon > 0.5)
	epsilon = 1.0 - epsilon
	FLIP_FLAG = true
end

alpha = 0.5*log((1.0-epsilon)/epsilon)


# E in {0,1}. Needs to be transformed to {-1,1}.
# 2*0-1 = -1 2*1 -1 =1
E = 2.0*E-1.0  # E = -1 if yp ==ya 1 otherwise

# Give higher weightage to W if yp != ya, i.e., when E == 1
# W = e^(-alpha) if yp == ya. W = e^(alpha) if yp != ya

W .*= exp(alpha.*E)
# Normalize W
W ./= sum(W)

#println(size(W));println(alpha);println(FLIP_FLAG);
return (W, alpha, FLIP_FLAG)

end
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
function getNewDataSet(weights::Array{Float64},numSamples::Int64)

# 	getNewDataSet(weights)
# 	takes as input:
#	1  weights: This is a m entry column vector which has values in (0,1). The value in weights(i)
#		correspond to the weight of the ith element in the base Data.
#		m = number of rows in base Data matrix.
#	2. numSamples is the number of samples to be drawn
#	Output:
#	A matrix D consisting of index to samples drwan with replacement from the base Data set.
#	D is a numSamples x 1 column vector
#		Resamples with replacement with probability of sample proportional to weight of data.

 if (numSamples <= 0)
 	error("Number of Samples is a positive Integer")
 end
 
 W = cumsum(weights)

 W ./= maximum(W) # sums up to 1
 
 new_Indices = Int64[]
 dice = rand(numSamples)
 for j =1:numSamples
	idx = findfirst(z->z > dice[j],W)
    push!(new_Indices, idx)
 end

 return new_Indices
end
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
function getThresholdAB( Prb, yk, weights)
  

  numRows = size(Prb,1)
  if size(weights,1) != numRows
  	error("Incompatible sizes")
  end
  res = 0.005
  R = [0.0:res:1.0]
  minEps = 10000.0
  threshold = 0.0
  
  for t in R 
  	epsilon = 0.0
  	yp = [ Prb[j] > t? 1:0 for j = 1:numRows ]
  	# Calculate epsilon
  	for k =1:numRows
   		epsilon += yp[k] != yk[k]? weights[k]:0.0
  	end
  	if epsilon < minEps
  		minEps = epsilon
  		threshold = t
  	end
  end
 
  yp = [Prb[k] > threshold ?1:0 for k =1:numRows]
  
  return (threshold, yp)
end
#=------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------  	
function adaPredictNB(trnSet::Array{Float64}, trainResults::Array{Any,2}, )

 Alphamat = vec(trainingParams[4,:])
 #-------------------------------------------------------------	
 # Remember [m,n] = size(Dt); that hasn't changed
 #-------------------------------------------------------------
 ProbMat = zeros(Float64, numRows,modelSize)
 cY = zeros(Int64, numRows,modelSize)
 yFinal = zeros(Int64, numRows) # Predicted outcome
 E = zeros(Float64, numRows) # Error
 for k = 1:modelSize
 	# Let each predictor predict and store the result as a column of ProbMat
 	(_,_, predProb) = predictNB(trnSet, trainingParams[1,k], trainingParams[2,k])
 	ProbMat[:,k] = predProb[:,2]
 	
 	# if flipFlag is set change the swap the probabilities							
 	if trainResults[5,k]			# Check to see if flipFlag is set
 		ProbMat[:,k] = broadcast(-,1.0,ProbMat[:,k])
 	end 										
 	cY[:,k]= [ProbMat[z,k] > trainingParams[3,k]?1:-1 for z = 1:numRows]
 end
 # Final predictions
 yFinal = sign(cY*Alphamat)  # Final predictions are in {-1,1}
 yFinal = dix((yFinal+1),2)  # Final predictions are in {0,1}
  		 	
 # Test error values
 E = [yFinal[z] != trnOut[z]?1.0:0.0 for z =1:numRows]
 err[iter] = sum(E)*100.0/numRows
=#
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
function  getNewWeightsSW(yp::Array{Int64},ya::Array{Int64}, weights::Array{Float64})

# function W = getNewWeightsSW(yp,ya,weights) -- selected weighting for 1
# Returns: (alpha, W)
# Assigns a weight to each data element based on whether the classifier has 
# classifier has predicted it accurately or not.
# Inputs:
#	yp is a num_data x 1 column vector. It is the predicted class of the classifier.
#	ya is a num_data x 1 column vector. It is the actual class.
# 	yp, ya in {0,1}
#	W is a num_data x 1 column vector. W(i) is the old weight for the row i in the base data set.
#	idx is the indices of the items from the base data set that were selected for training
# Outputs:
#	W is a num_data x 1 column vector. W(i) is the new weight for the row i in the base data set.
#

FLIP_FLAG = false


W = vec(weights./sum(weights))
num_data = length(yp) # Number of outcomes = number of data elements in base set

if length(ya) != num_data
	error("Mismatch in lengths of predicted and actual outcome vectors")
end

# Compute alpha = 0.5*ln((1-epsilon)/epsilon) epsilon is the error rate

E = zeros(Float64,num_data)

for k =1:num_data
	if (ya[k] == yp[k])
		continue
	end
	if (ya[k] == 1)
		E[k] = 1.5
	else
		E[k] = 1.0
	end		
end
#E = [ yp[k] != ya[k]?1.0:0.0 for k = 1:num_data] # E = 0 if yp == ya, E = 1 if yp != ya.

epsilon = 0.0
# How does this work:
# We always use the Base Training Set to compute the error.
# So error is proportional to the weight of each data element in the BTS.
for k = 1:num_data
	if (yp[k] != ya[k])
		epsilon += W[k]
	end
end


if (epsilon > 0.5)
	epsilon = 1.0 - epsilon
	FLIP_FLAG = true
end

alpha = 0.5*log((1.0-epsilon)/epsilon)


# E in {0,1}. Needs to be transformed to {-1,1,2}.
# 2*0-1 = -1 2*1 -1 =1
E = 2.0*E-1.0  # E = -1 if yp ==ya 1 otherwise

# Give higher weightage to W if yp != ya, i.e., when E == 1
# W = e^(-alpha) if yp == ya. W = e^(alpha) if yp != ya

W .*= exp(alpha.*E)

# Normalize W
W ./= sum(W)

#println(size(W));println(alpha);println(FLIP_FLAG);
return (W, alpha, FLIP_FLAG)

end
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

