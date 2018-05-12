#!/usr/bin/julia -q
# Version 0.0
# Date: 09-06-15 at 08:23:02 PM IST
# Version 0.0
# Date: 09-06-15 at 06:53:55 PM IST
# Version 4.0
# Date: 10-05-15 at 11:00:00 PM IST

# Command line arguments are stored in the variable ARGS
#--------------------------------------------------------
println("Loading Libraries ...")
include("Library/general.jl")
include("Library/Bayesian.jl")
include("Library/adaboost.jl")
#--------------------------------------------------------
filename="ami"

println("Reading dataset: $filename")

data_file 		= @sprintf("%s_data.jla",filename)
outcome_file 	= @sprintf("%s_y.jal",filename)

data = readArray(data_file)
y = readArray(outcome_file)



# metaData needs to be created
metaData = makemetadata(data)
(data, LUT) = encodeX(data,metaData)
uniqY = sort(unique(y))
numClasses = length(uniqY)
#--------------------------------------------------------
#		
#--------------------------------------------------------
maxIters = 30
maxModel = 50
sampleFraction = 0.1

errMatrixTrain = zeros(Float64,maxModel)
errMatrixTest = zeros(Float64,maxModel)


tic()
for modelSize = 1:maxModel
#---------------------------------------
	err = zeros(Float64,maxIters)
	errCV = zeros(Float64,maxIters)
	for iter = 1:maxIters
	
	#---------------------------------------	
		(trnSet, trnOut, cvSet, cvOut) = getCVdata(sampleFraction,data,y)
		(numRows, numCols) = size(trnSet)
		weights = ones(Float64,numRows)./float64(numRows)
		# Store for the training parameters for each iteration
		#isdefined(:trainingParams) || trainingParams = cell(5,modelSize)
		
		trainingParams = cell(5,modelSize)
		# [1,k] = yLogProbs
		# [2,k] = nodeLogProbs
		# [3,k] = threshold
		# [4.k] = Alpha
		# [5,k] = flipFlag
		
		for k =1:modelSize
			@printf("Model size: %3d, Iteration (%3d) building model: %3d \r",modelSize, iter,k)
			#-------------------------------------------------------------------------------------------------
			#	This entire thing can be put in a function ...
			#	BEGIN adaTrainNB
			#-------------------------------------------------------------------------------------------------
			didx = getNewDataSet(weights,numRows)
			Dk = trnSet[didx,:]
			yk = trnOut[didx,:]
			
			
			#-----------------------------------------
			#	Train using Naive bayes
			#----------------------------------------
			
			(trainingParams[1,k],trainingParams[2,k] ) = trainNB(Dk,yk,numClasses, LUT) # Store the training parameters
			
			#----------------------------------------
			# ok test against base data set
			#---------------------------------------
			
			(classIndex ,_ ,ProbVal) = predictNB(trnSet, trainingParams[1,k],trainingParams[2,k])
			yp = uniqY[classIndex]
			
			
			#(trainingParams[3,k], yp) = getThresholdAB( ProbVal[:,2], trnOut, weights)
			
			#-----------------------------------------------------------------------
 			# Get new weights, compute the current alpha and flipflag
 			#-----------------------------------------------------------------------
 			(weights,trainingParams[4,k],trainingParams[5,k]) = getNewWeights(yp,trnOut,weights)
			
			#-------------------------------------------------------------------------------------------------
			#	This entire thing can be put in a function ...
			#	END adaTrainNB
			#-------------------------------------------------------------------------------------------------
			
		end # Training
		
		
		# CALCULATE TRAINING ERROR
		#--------------------------------------------------------------------------------------------
		# 	This entire thing contains the predict part
		#	BEGIN adaPredictNB
		#--------------------------------------------------------------------------------------------
		Alphamat = zeros(Float64,modelSize,1)
		for k =1:modelSize
			Alphamat[k,1] = trainingParams[4,k]
		end
		
		
		#-------------------------------------------------------------	
 		# Remember [m,n] = size(Dt); that hasn't changed
 		#-------------------------------------------------------------
 		ProbMat = zeros(Float64, numRows,modelSize)
 		cY = zeros(Float64, numRows,modelSize)
 		yFinal = zeros(Int64, numRows) # Predicted outcome
 		E = zeros(Float64, numRows) # Error
 		
 		
 		for k = 1:modelSize
 			# Let each predictor predict and store the result as a column of ProbMat
 			(classIndex,_, predProb) = predictNB(trnSet, trainingParams[1,k], trainingParams[2,k])
 			cY[:,k] = float64(uniqY[classIndex])
 			
 			# if flipFlag is set change the swap the probabilities							
 			if trainingParams[5,k]			# Check to see if flipFlag is set
 				cY[:,k] = 1.0 - cY[:,k]
 			end 										
 			
 			#cY[:,k]= [ProbMat[z,k] > trainingParams[3,k]?1:-1 for z = 1:numRows]
 		end
 		cY = 2.0.*cY .-1.0
 		# Final predictions
 		yFinal = sign(cY*Alphamat)  # Final predictions are in {-1,1}
 		yFinal = div((yFinal+1),2) 	# Final predictions are in {0,1}
  		
 		# Test error values
 		E = [yFinal[z] != trnOut[z]?1.0:0.0 for z =1:numRows]
 		err[iter] = sum(E)*100.0/numRows
 		#------------------------------------------------------------------------
 		# Clculate test errors
 		#
 		#------------------------------------------------------------------------
 		numRows = size(cvSet,1)
 		ProbMat = zeros(Float64, numRows,modelSize)
 		cY = zeros(Float64, numRows,modelSize)
 		yFinal = zeros(Int64, numRows) # Predicted outcome
 		E = zeros(Float64, numRows) # Error
 	
 		
 		for k = 1:modelSize
 			(classIndex,_, predProb) = predictNB(cvSet, trainingParams[1,k], trainingParams[2,k])
 			cY[:,k] = float64(uniqY[classIndex])
 			
 			# if flipFlag is set change the swap the probabilities							
 			if trainingParams[5,k]			# Check to see if flipFlag is set
 				cY[:,k] = 1.0 - cY[:,k]
 			end 										
 			
 			#cY[:,k]= [ProbMat[z,k] > trainingParams[3,k]?1:-1 for z = 1:numRows]
 		end
 		cY = 2.0.*cY .-1.0
 		# Final predictions
 		yFinal = sign(cY*Alphamat)  # Final predictions are in {-1,1}
 		yFinal = div((yFinal+1),2) 	# Final predictions are in {0,1}
  		
 		# Test error values
 		E = [yFinal[z] != cvOut[z]?1.0:0.0 for z =1:numRows]
 		errCV[iter] = sum(E)*100.0/numRows
 		
 		#----------------------------------------------------------------------- 		
 	end # Each iteration
 	errMatrixTrain[modelSize] = mean(err)
 	errMatrixTest[modelSize] = mean(errCV)
end
toc()

println("Plotting ...")
using Gadfly
pic = plot(Theme(background_color=color("white")), Guide.xlabel("Model size"), Guide.ylabel("Error (%)"),
			Guide.manual_color_key("Adaboost with Naive Bayes", ["Train Error", "Test Error"], ["#007FFF", "#FF6600"]),
			layer(x=[1:maxModel], y=errMatrixTrain, Geom.line, Theme(default_color=color("#007FFF"))),
			layer(x=[1:maxModel], y=errMatrixTest,  Geom.line, Theme(default_color=color("#FF6600")))
			)
draw(PNG("testAdaboost-NB_NT.png", 24cm, 15cm), pic)
			
			
			
			
			
			
			
			
			
			
