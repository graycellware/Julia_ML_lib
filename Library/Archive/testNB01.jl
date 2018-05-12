#!/usr/bin/julia -q
# Version 1.0
# Date: 09-05-15 at 07:46:52 PM IST

# Command line arguments are stored in the variable ARGS
#
if (length(ARGS) == 0)
	println("Usage: testNB.jl <.mat file>")
end


num_files = length(ARGS)
SUCC_COUNT = 0
ERR_FLAG = false
FLS = readdir()
y =0; data =0;
for k = 1:num_files
	
	fileparts = match(r"^\s*(\S+?)\.(.+?)$", ARGS[k])
	if (fileparts == nothing)
		ERR_FLAG = true
		continue
	end
	filename=fileparts.captures[1]
	extension=fileparts.captures[2]
	if (!ismatch(r"mat"i,extension))
		@printf("Extn mismatch: Unable to process %s\n", ARGS[k])
			ERR_FLAG = true
			continue
	end
	# ok we have a .mat file
	file_name = ARGS[k]
	try
		run(`mat2csv.m "$file_name"`) # This should create two files
	catch
		@printf("MAT2CSV Conversion Error: %s\n", ARGS[k])
		ERR_FLAG = true
		continue
	end
		
	try
		regstr = "$(filename)"
		regexstr = Regex(regstr,"i")
		target_files = [match(regexstr,FLS[l]) for l = 1:length(FLS)]
		
		idx = Int64[]
				
		[target_files[k] == nothing?continue:
							push!(idx,k) for k =1:length(target_files)]
	catch
		@printf("Unable to find data,outcome files: %s\n", ARGS[k])
		ERR_FLAG = true
		continue
	end
	if(isempty(idx))
		ERR_FLAG = true
		continue
	end
	# idx is not empty
	target_files = FLS[idx]
	
	idx = Int64[]
	# Get the data file
	for kk = 1:length(target_files)
		if ismatch(r"_data.csv"i,target_files[kk])
			push!(idx,kk)
		end
	end
	if (isempty(idx))
		@printf("No data files %s\n", ARGS[k])
		ERR_FLAG = true
		continue
	end
	try
		data = readdlm(target_files[idx[1]],',')
	catch
		@printf(" Error reading data file: %s\n", ARGS[k])
		ERR_FLAG = true
		continue
	end
	
	idx = Int64[]
	# Get the data file
	for l = 1:length(target_files)
		if ismatch(r"_y.csv"i,target_files[l])
			push!(idx,l)
		end
	end
	
	if (isempty(idx))
		@printf("No outcome files %s\n", ARGS[k])
		ERR_FLAG = true
		continue
	end
	try
		y = readdlm(target_files[idx[1]],',')
	catch
		@printf("Error reading outcome file: %s\n", ARGS[k])
		ERR_FLAG = true
		continue
	end
	SUCC_COUNT++
	# Successfully read both data and outcome files
end

# Next, let us 

exit(0)







