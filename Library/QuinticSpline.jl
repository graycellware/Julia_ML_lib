#
# Archived version: 1 on 09-06-15 at 06:55:00 PM IST
function getCubicSpline(data::Array{Float64}, alpha::Float64, beta::Float64)
# data is a m x 2 matrix in which the 1st column
# is assumed to contain the x values and
# the second column contains the corresponding y values
# 
# Given y = f(x), it returns f'(x) and f"(x) after spline fitting

  (N, ncols) = size(data)
  	
  if ncols < 2
	error("insufficient number of columns in input data")
  end

  X = sortrows(data,by=x->x[1])
  
  
  Dely = zeros(Float64,(N-1))
  TDM = zeros(Float64, N, N)
  
  for k = 1:(N-1)
  	Dely[k] = (X[(k+1),2] - X[k,2])/(X[(k+1),1] - X[k,1])
  end
  
  #--------------- Create the Tridiagonal matrix -----------
  for k = 2:(N -1)
  	TDM[k,k-1] = X[k,1] - X[k-1,1]
  	TDM[k,k+1] = X[k+1,1] - X[k,1]
  	TDM[k,k] = 2*(TDM[k,k-1] + TDM[k,k+1])
  end
  TDM[1,1] = (X[2,1] - X[1,1])/3.0; TDM[1,2] = (X[3,1] - X[2,1])/6.0
  TDM[N,(N-1)] = (X[(N-1),1] - X[(N-2),1])/6.0; TDM[N,N] = (X[N,1] - X[(N-1),1])/3.0
  #---------------------------------------------------------
  #--------------- Create the V column matrix---------------
  Vmatrix = zeros(Float64,N)
  for j = 2:(N-1)
  	Vmatrix[j] = Dely[j] - Dely[j-1]
  end
  Vmatrix[1] = (Dely[1] - alpha)
  Vmatrix[N] = (beta - Dely[(N-1)])
  #---------------------------------------------------------
  M = zeros(Float64,N,2)
  M[:,1] = 6.0*pinv(TDM)*Vmatrix
  
  for k =1:(N-1)
  	M[k,2] = (-2.0*M[k,1] - M[(k+1),1])*(X[k+1,1] - X[k,1])/6.0 + Dely[k]
  end
  
  M[N,2] = (2.0*M[N,1] - M[(N-1),1])*(X[N,1] - X[N-1,1])/6.0 + Dely[N-1]
  return M
end

#-----------------------------------------------------------
function getCubicSpline(data::Array{Float64})
	return getCubicSpline(data,0.0,0.0)
end

  
