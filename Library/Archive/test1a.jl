#!/usr/bin/julia

println("Importing libraries ...")
#using Distributions
using DataFrames
using Gadfly
include("ent.jl")
println("Done")

println("Creating random data ...")
# Create a set of random data
x1 = zeros(5000,1)
x2 = zeros(2500,1)
x3 = zeros(3500,1)


mu1 = 1.3; mu2 = 4.3; mu3 = 5.6
sig1 = 1.0; sig2 = 1.3; sig3 = 1.4

# Generate the data
x = [	rand!(Normal(mu1,sig1),x1);
		rand!(Normal(mu2,sig2),x2);
		rand!(Normal(mu3,sig3),x3); ]

println("Done")
# Get the kde
# Returns a range as K[:,1], and kde as K[:,2]

println("Create KDE ...")
K = kde(x)
println("Done")
# Get the corresponding cdf
println("Generate CDF ...")
C = kde2cdf(K)
println("Done")
# Finally get the piecewise-linear cdf approximations
println("Generate piecewise linear CDF ...")
A = plcdf(K)
println("Done")
exit
