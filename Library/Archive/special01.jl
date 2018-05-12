function myhypot(x,y)
# Version 1.0
# Date: 09-05-15 at 12:52:16 AM IST
  z = abs(x)
  w = abs(y)
  maxVal = max(z,w)
  minVal = min(z,w)
  r = minVal/maxVal
  if(w*z <= 1e-5)
    return maxVal*(1+r/2)
  end
  return maxVal*sqrt(1+r*r)
end


