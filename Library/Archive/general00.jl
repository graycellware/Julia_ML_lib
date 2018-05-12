#!/usr/bin/julia -1
# Version 0.0
# Date: 26-05-15 at 07:21:48 AM IST

using Match

function myhypot(x,y)
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
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
function writeArray(filename::String, A::Array, R::DataType)
#= This is how it works ...
We write a set of meta information in the file the same way MATLAB does. We also store the array as a column, and use 'reshape'' to restructure it because we know the size from the metadata.
=#
 FN = match(r"^(.+)\..+",filename)
 
 if (FN == nothing)
 	outFile = string(filename,".","jla")
 else
 	outFile = string(FN.captures[1],".","jla")
 end

 f = open(outFile, "w")

 write(f,"# Julia array store version 0.1\n")
 write(f, @sprintf("# Dim:%s\n", string(ndims(A))))
 write(f, @sprintf("# Size:%s\n", string(size(A))))
 write(f, @sprintf("# Type:%s\n", string(R)))
 
 for z in 1:length(A)
 	write(f,@sprintf("%s\n",A[z]))
 end
 close(f) # write back to disk
 return 1
end

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
function readArray(filename::String)

 FN = match(r"^(.+)\..+",filename)
 
 if (FN == nothing)
 	inFile = string(filename,".","jla")
 else
 	inFile = string(FN.captures[1],".","jla")
 end

 f = open(inFile, "r")
 # file has at least 4 lines
 
 goodStuff = readall(f)
 close(f)
 lines = matchall(r"(.+?)\n",goodStuff)
 #------------------
 A_dims = match(r"^\s*#\sDim:(.+)"i, chomp(lines[2])).captures[1]
 
 if A_dims == nothing
	error("Invalid Julia Array File: No/Invalid dimension entry in line 2")
 end
 #--------------------
 A_size = match(r"^\s*#\sSize:(.+)"i, chomp(lines[3])).captures[1]
 
 if A_size == nothing
	error("Invalid Julia Array File: No/Invalid size entry in line 3")
 end

  #--------------------
 A_type = match(r"^\s*#\sType:(.+)"i, chomp(lines[4])).captures[1]
 
 if A_type == nothing
	error("Invalid Julia Array File: No/Invalid type entry in line 4")
 end
  #--------------------
 
  # Parse the dimensions
   num_dims = parseint(A_dims) 
 
  # We now have the number of dimensions
  # Size is going to be of the form (x_1, ..., x_dim)
  # Parse size
  size_Array = zeros(Int64,num_dims)
  numels =1
  # Get the sizes ...
  if (num_dims > 0)
  	sizes = matchall(r"[0-9]+",A_size)
  	dimensions = length(sizes)
  	
  	if (dimensions != num_dims)
  		error("Ill-formed Array")
  	end
  	# Expected number of entries = product of all elements of size_Array
  	for q = 1:dimensions
  		size_Array[q] = parseint(sizes[q])
  		numels *= size_Array[q]
  	end
  end
  
  target = Int64[]
  
  if (A_type in ["Float64", "Float32", "Real" ])
  	for k = 1:numels
  		target = float64(target)
  		push!(target,parsefloat(chomp(lines[k+4])))
  	end
  else
  	for k = 1:numels
  		push!(target,int64(parsefloat(chomp(lines[k+4]))))
  	end
  end
  
  b = tuple(size_Array...)
  return reshape(target,b)
end
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
function drawGraph(incidenceMatrix::Array{Int64}, filename::ASCIIString)
	
	numCols = size(incidenceMatrix,1)
	if ((filename==nothing) || (length(filename) == 0))
		graphFile = "graphView.tex"
	else
		graphFile = string(filename,".tex")
	end
		
	of = open(graphFile, "w")
	RET_MSG = graphFile

Header = """\\documentclass[tikz,margin=5pt]{standalone}
\\usetikzlibrary{graphs,graphdrawing,arrows}
\\usegdlibrary{force}
\\begin{document}
\\tikz[spring layout, node distance=25mm,>=latex']{
"""
	write(of,Header)
	
	# Writing Nodes
	for k = 1:numCols
		write(of, @sprintf("\\node (%d) {column\\_%s};\n",k,string(k)))
	end
	write(of, "\\draw\n")
	# Writing Edges
	for k = 1:(numCols-1), l = (k+1):numCols
		if(incidenceMatrix[k,l] == 1)
			write(of, @sprintf("(%d) edge (%d)\n", k, l))
		end
	end
	write(of,";\n")
	write(of,"}\n\\end{document}")
	close(of)
	return RET_MSG
end
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
function csv2jla(fileName::ASCIIString, dataType::DataType)

# Input can be of either form:
# 1. just the filename without the csv extension
# 2. Filename with the csv extension

 fileparts = match(r"^(.+)\.(.+)$", fileName)
 
 if (fileparts == nothing) # Filename probably is without an extension
 	if (isempty(fileName))
 		error("No filename given")
 	end
 	
 	csvFile = string(fileName,".csv")
 	CSVFile = string(fileName,".CSV")
 	if(!isfile(csvFile))
 		if(!isfile(CSVFile))
 			error("File not found")
 		else
 			file_name = CSVFile
 		end 		
 	else
 		file_name = csvFile
 	end
 	target = string(fileName,".jla")	
 else
 	if (lowercase(fileparts.captures[2]) == "csv")
 		file_name = fileName
 		if(!isfile(file_name))
 			error("File not found")
 		end
 		target = string(fileparts.captures[1],".jla")	
 	else
 		csvFile = string(fileparts.captures[1],".csv")
 		CSVFile = string(fileparts.captures[1],".CSV")
 		
 		if(!isfile(csvFile))
 			if(!isfile(CSVFile))
 				error("File not found")
 			else
 				file_name = CSVFile
 			end 		
 		else
 			file_name = csvFile
 		end
 		target = string(fileparts.captures[1],".jla")	
 	end
 end
  	
 data = readdlm(file_name,',')
 R = typeof(data[1,1]) # take a sample
 
 if (dataType == R)
 	writeArray(target,data,dataType)
 else
 	@match  dataType begin
 		"Float64" 	=> writeArray(target,float64(data),Float64)
 		"Float32" 	=> writeArray(target,float32(data),Float32)
 		"Int64" 	=> writeArray(target,int64(data),Int64)
 		"Int32" 	=> writeArray(target,int32(data),Int32)
 		_			=> writeArray(target,data,ASCIIString)
 	end
 end

end
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

