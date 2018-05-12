
# Archived version: 1 on 26-05-15 at 07:22:12 AM IST


using Distributions

function Qkde(x::Array{Float64})
	
	# Computes the Kernel Density Estimate of x
	# Using a Gaussian Radial Basis function
	# returns a N x 2 array
	# the first column gives the range
	# Column 2 gives the kde value
	lenX = length(x)
	stdX = std(x)
	res = 2.0*stdX/10.0
	minX = minimum(x) -3.0res
	maxX = maximum(x) +3.0res
	R = (minX:res:maxX)
	kdeVal = zeros(Float64,length(R))
	for k = 1:lenX
		kdeVal += pdf(Normal(x[k],res),R)
	end
	kdeVal ./= lenX
	return [R kdeVal]
end

function Qkde(x::Array{Float64}, res::Float64)
	
	# Computes the Kernel Density Estimate of x
	# Using a Gaussian Radial Basis function
	# returns a N x 2 array
	# the first column gives the range
	# Column 2 gives the kde value
	lenX = length(x)
	minX = minimum(x) -3.0res
	maxX = maximum(x) +3.0res
	stdX = std(x)
	sig = stdX*((4.0/3.0lenX)^0.2)
	R = (minX:res:maxX)
	kdeVal = zeros(Float64,length(R))
	for k = 1:lenX
		kdeVal += pdf(Normal(x[k],sig),R)
	end
	kdeVal ./= lenX
	return [R kdeVal]
end

function kde2cdf(x::Array{Float64})
	# Given a kde, computes the corresponding
	# cdf
	# returns a N x 2 array
	# the first column gives the range
	# Column 2 gives the cdf value
	num_rows, num_cols = size(x)
	if (num_cols != 2)
		error("inpute needs to be an N x 2 array")
	end
	res = x[2,1] - x[1,1]
	cdf = zeros(Float64, num_rows)
	# We should actually integrate
	cdf = cumsum(x[:,2]).*res
	cdf ./= maximum(cdf)
	return [x[:,1] cdf]
end

# Piecewise linear approximation of cdf

function plcdf(x::Array{Float64})
	# Given a kde, computes a piecewise
	# linear approximation for the cdf
	
	
	num_rows,num_cols = size(x)
	if (num_cols != 2)
		error("inpute needs to be an N x 2 array")
	end
	
	res = x[2,1] - x[1,1]
	cdf = zeros(Float64, num_rows)
	cdf = cumsum(x[:,2]).*res
	cdf ./= maximum(cdf)
	
	# Compute the Jacobian
	inflexions = Int64[1] 	# initialize Int64 Array
							# by default includes the first point
	absdifftan = zeros(Float64,num_rows)
	for k = 2:(num_rows-1)
		# We will use this to check slope at every point
		# both above and below
		# if the change in slope is greater than a threshold,
		# add the point to the list of inflexions
		# it indicates an extremum
		slope_below = (cdf[k] - cdf[k-1])/res
		slope_above = (cdf[k+1] - cdf[k])/res
		absdifftan[k] = abs(atan((slope_above -slope_below)/(1+(slope_above*slope_below)))) # Difference in radians
	end
	slopes = zeros(Float64,num_rows)
	slopes = sort(absdifftan, rev=true)
	# Lets take the top 70%
	k = iround(num_rows*0.3)
	angle = slopes[k];
	
	for k = 1:num_rows
		if(absdifftan[k] >= angle)
			push!(inflexions, k)
		end
	end
	
	# Next, we need the point that is closes to being 1 from the left
	err = 0.5*(1-cdf[inflexions[end]])
	k = findfirst(z->z > (1-err),cdf)
	push!(inflexions, k)
	# by default, inflexions includes the last point of x
	push!(inflexions, num_rows)
	sort!(inflexions)
	cdfVals = zeros(Float64, length(inflexions))
	cdfVals = cdf[inflexions]
	return 		[ x[inflexions] cdfVals]
end

function getprobVal(x::Float64, res::Float64, Arr::Array{Float64})
	# x is value within the range of Arr
	# Arr is a piecewise linear approximation of
	# the cdf (any array representation of the cdf is a piecewise linear approximation)
	# returns the probability of the interval [x-0.5res, x + 0.5res]
	if ((x < Arr[1,1]) || (x > Arr[end,1]))
		return 0.0
	end
	num_rows = size(Arr,1)
	del=0.5*res
	
	if((x+del) > Arr[end,1])
		y0 = 1 - Arr[(end-1),2]
		x0 = Arr[end,1] - Arr[(end-1),1]
		p0 = (y0/x0)*((x-del)-Arr[(end-1),1]) + Arr[(end-1),2]
		return 1 - p0
	end
	
	if((x-del) < Arr[1,1])
		y1 = Arr[2,2]
		x1 = Arr[2,1] - Arr[1,1]
		p1 = (y1/x1)*((x+del)-Arr[1,1])
		return p1
	end
	
	idx1 = findfirst(z->z >= (x+del), Arr[:,1])
	y1 = Arr[idx1,2] -Arr[(idx1-1),2]
	x1 = Arr[idx1,1] -Arr[(idx1-1),1]
	probVal1 = (y1/x1)*((x+del)-Arr[(idx1-1),1]) + Arr[(idx1-1),2]
	
	
	idx0 = findfirst(z->z >= (x-del), Arr[:,1])
	y0 = Arr[idx0,2] -Arr[(idx0-1),2]
	x0 = Arr[idx0,1] -Arr[(idx0-1),1]
	probVal0 = (y0/x0)*((x-del)-Arr[(idx0-1),1]) + Arr[(idx0-1),2]
		
	return (probVal1-probVal0)
end	

function getprobVal(x::Float64, Arr::Array{Float64})
	# x is value within the range of A
	# A is a piecewise linear approximation of
	# the cdf (any array representation of the cdf is a piecewise linear approximation)
	# returns the probability of the interval [x-0.5res, x + 0.5res]
	
	# This is interesting
	# get the range
	num_rows = size(Arr,1)
	minVal = Arr[end,1]+10.0
	for k =2:num_rows
		if((Arr[k,1]-Arr[(k-1),1]) < minVal)
			minVal = Arr[k,1]-Arr[(k-1),1]
		end
	end
	res = minVal/4.0	
	return getprobVal(x,res,A)
end



