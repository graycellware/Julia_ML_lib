#!/usr/bin/julia
# Archived version: 1 on 09-05-15 at 12:52:22 AM IST

#-------------------------------------------------------
println("Importing libraries ...")
#using Distributions
using DataFrames
using Gadfly
include("ent.jl")
println("Done")
#-------------------------------------------------------
println("Creating random data ...")
# Create a set of random data
x1 = zeros(5000,1)
x2 = zeros(2500,1)
x3 = zeros(3500,1)


mu1 = -1.3; mu2 = 4.3; mu3 = 5.6
sig1 = 1.0; sig2 = 0.9; sig3 = 1.4

# Generate the data
x = [	rand!(Normal(mu1,sig1),x1);
		rand!(Normal(mu2,sig2),x2);
		rand!(Normal(mu3,sig3),x3); ]

println("Done")
#-------------------------------------------------------
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
y = rand()*(A[end,1] - A[1,1]) + A[1,1]

println(y)
v1 = getprobVal(y,C)
println(v1)

v1 = getprobVal(y,A)
println(v1)



println("Done")



#-------------------------------------------------------
println("Plotting ...")

df1 = DataFrame(x=K[:,1], y=K[:,2], label="KDE")
df2 = DataFrame(x=C[:,1], y=C[:,2], label="Regular CDF")
df3 = DataFrame(x=A[:,1], y=A[:,2], label="Compressed CDF")

df = vcat(df1, df2, df3)

pic = plot(df, x="x", y="y", color="label", Geom.line,
			Geom.vline(color="black",size=0.1mm), xintercept=A[:,1],
			Theme(	panel_fill=color("#FFFFFF"),
					key_title_font="Century Schoolbook L",
					key_title_font_size=10pt,
					key_label_font_size=9pt,
					),
			Guide.xlabel("x"), Guide.ylabel("Probability"),
			Guide.title("Comparing Regular Vs Compressed CDF"),
         	Scale.color_discrete_manual("#FFBF00","#21ABCD", "#915C83"))



draw(SVG("myplot.svg", 19.4cm, 12cm),pic)
println("Done")
exit
