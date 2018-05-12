function getLabDist(RGB::Array{Float64}, refRow::Int64)
# Version 0.0
# Date: 17-05-15 at 03:14:09 PM IST

num_row,num_cols = size(RGB)
if (num_cols != 3)
	error("Not a valid RGB matrix")
end

if ((maximum(RGB) > 1.0) || (minimum(RGB) < 0.0))
	error("Not valid RGB ranges")
end

if (0 > refRow > 1)
	error("Invalid reference Row")
end

M = [2.04159 -0.56501 -0.34473;
	 -0.96924 1.87587 0.04156;
	 0.01344 -0.11836 1.01517 ]

Xn = 0.95047; Yn = 1.0; Zn = 1.08883;
xyz = *(RGB,M.')

alpha = 6.0/29.0
beta = 1.0/(3alpha^2)
gamma = 4.0/29.0
f(t) = (t > alpha^3?t^(1.0/3.0):beta*t + gamma)
Lab = zeros(Float64,size(xyz))

for k=1:num_row
	cratio = f(xyz[k,2]/Yn)
	Lab[k,1] = 116.0*cratio -16.0
	Lab[k,2] = 500.0*(f(xyz[k,1]/Xn) -cratio)
	Lab[k,3] = 200.0*(cratio - f(xyz[k,3]/Zn))
end

K1 = 0.045; K2 = 0.015; kL = 1.0;
dell = repmat(Lab[:,1],1,num_row)
delL = dell -dell.';
delA = repmat(Lab[:,2],1,num_row)
dela = delA -delA.'
delB = repmat(Lab[:,3],1,num_row)
delb = delB -delB.'
C1 = sqrt(delA.*delA + delB.*delB)
C2 = C1.'
delCab = C1 -C2
delHab = sqrt(dela.*dela + delb.*delb - (delCab.*delCab))
SL = 1.0; SC = 1.0 .+ (K1.*C1); SH = 1.0 .+ (K2.*C1)
Term1 = delL./(kL.*SL); Term2 = delCab./SC; Term3 = delHab./SH;
dist = sqrt(Term1.*Term1 + Term2.*Term2 + Term3.*Term3)
print()
return sortperm( vec(dist[:,refRow]))
end

