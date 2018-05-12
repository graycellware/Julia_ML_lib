#!/usr/bin/julia -q

 using Match
 if length(ARGS) == 0
	println("Usage: csv2jla [type] <.csv file> ")
	exit(-1)
 end
 R_FLAG =true
 if ARGS[1] in ["Float64", "Int64", "Float32", "Int32"]
 	R_FLAG = false
 	R = shift!(ARGS)
 end
 	
 
 for k =1:length(ARGS)
 	filename = match(r"^(.+)\.csv$"i, ARGS[k])
 	if filename == nothing
		error("Invalid CSV file")
 	end
 	include("general.jl")
 	filename = string(filename.captures[1],".jla")
 	data = readdlm(ARGS[k],',')
 	if R_FLAG
 		R = typeof(data[1,1])
 		writeArray(filename,data,R)
 	else
 		@match R begin
 			"Float64" 	=> writeArray(filename,float64(data),Float64)
 			"Float32" 	=> writeArray(filename,float32(data),Float32)
 			"Int64" 	=> writeArray(filename,iround(data),Int64)
 			"Int32" 	=> writeArray(filename,int32(iround(data)),Int32)
 			_			=> writeArray(filename,data,ASCIIString)
 		end
 	end
 end
exit()
