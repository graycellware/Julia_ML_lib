#!/usr/bin/julia
# Version 1.0
# Date: 09-05-15 at 01:06:40 PM IST



function makemetadata(X::Array{Int64})
#		metaData : n x 3 matrix 
#		Column 1: ith entry gives minimum value that column i of X can take (assume all 0)
# 		Column 2: ith entry gives maximum value that column i of X can take
#		Column 3: ith entry gives number of values that column i of X can take
#			We assume rows take values between 0 and k(i)-1 
#			where k(i) is the entry in column 2, row i of metaData.
	n = size(X,1)	# Number of columns of X
	metaData = zeros(Int64,n,3)
	metaData[:,2] = maximum(X,1) - minimum(X,1)
	metaData[:,3] = metaData[:,2] .+ 1
	return metaData
end


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
	
	LUT = zeros(Int64, num_nodes,3)
	
	Indx = 1
	for column_in_X = 1:num_cols
		k = metaData[column_in_X,3]		# k is the number of values that the 
										# corresponding column of X can take.
		LUT[rowIndx:(Indx+k-1), 1] = [0:(k-1)] .+ (column_in_X*kmax) # Coded value
		LUT[rowIndx:(Indx+k-1), 2] = ones(Int64,k).*k 	# store the max values per column
		LUT[rowIndx:(Indx+k-1), 3] = ones(Int64,k).*column_in_X 	# store the column numbers
		Indx += k
	end
	
	# Ensure that the columns of X range between 0 and (k-1)
	X_Col_Min = minimum(X,1)
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
	endfor
	# Now the entries in X are the corresponding indices in LUT(1,:)
	 return [X LUT]
end


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

	Z = zeros(Int64, size(X))
	Z = 0 .+ X
	return encodeX!(Z, metaData)
end


function trainNB(X::Array{Int64},y::Array{Int64}, LUT::Array{Int64})

	(num_rows, num_cols) = size(X)
	
	target_classes = sort(unique(y))
	num_classes = length(target_classes)
	##------------------------------------------------------------------------------
	# Define the other output matrices
	##------------------------------------------------------------------------------
	yLogProbs = zeros(Float64, num_classes)
	nodeLogProbs = zeros(Float64, num_classes,num_nodes) # 2-D matrix
	
	
	##------------------------------------------------------------------------------
	# Populate nodeLogProbs
	##------------------------------------------------------------------------------
	for y_class = 1:num_classes
		yidx = find(z->z == target_classes(y_class), y)
		
		#--------------------------------------------
		# Compute yLogProbs (with Lapalace estimator)
		#--------------------------------------------
		yLogProbs[y_class] = log((length(yidx)+1.0)/(num_rows+num_classes))
		
		#--------------------------------------------
		# Calculate count of node given y_class
		#--------------------------------------------
				
		for node_value = 1:num_nodes
			nodeLogProbs[y_class,node_value] += 
						length(find(z->z == node_value, X[yidx,:] ))
		end
	end
	#
	#------------------------------------------------------------------------------
	# Apply Lapalce estimators to nodeLogProbs
	# numerator is easy -- just add 1
	# denominator is K + num_classes*(number of values column can take)
	# The number of values the column can take is already factored in LUT(2,:)
	##------------------------------------------------------------------------------
	denom = LUT[:,2].*num_classes .+ num_rows
	nodeLogProbs = log((nodeLogProbs+1.0)./repmat(denom,1,num_classes))
	##------------------------------------------------------------------------------
end


