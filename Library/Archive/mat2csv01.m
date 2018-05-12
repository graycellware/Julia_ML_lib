#!/usr/bin/octave -q
# Version 1.0
# Date: 09-05-15 at 01:06:33 PM IST

arg_list = argv ();

if (nargin == 0)
	printf("Usage: %s <.mat file>",program_name());
	exit;
endif

#  [dir, name, ext, ver] = fileparts (filename)

for k = 1:nargin
    [~, name, ext, ~] = fileparts(arg_list{k});
  	load(arg_list{k});
  	# Find out the variables
  	C = who('-file',arg_list{k});
  	 	
  	for l = 1:length(C)
  		out_file = [name,'_',C{l},'.csv'];
  		
  		evalstr = sprintf("csvwrite('%s',%s)",out_file,C{l});
  		printf("%s\n", evalstr);
  		eval(evalstr);
  	endfor
endfor
exit;	
