# This file contains the implementation of Random Forests using m-ary CART
# Version 0.0
# Date: 14-08-15 at 07:53:37 PM IST
# The m-ary trees do not

#using Switch

#const (ne, lt, le, eq, gt, ge) = (0,1,2,3,4,5)

function giniImpurity(y::Array{Int64},uniqY::Array{Int64})
# this function computes gini impurity.
	m = length(y)
	score = 0.0
	for val in uniqY
		score += (length(find(z->z == val,y))/m)^2
	end
	return 1.0 -score
	
end
		
abstract CART
type treeCell <: CART
	parentId::Int64
	nodeId::Int64
	numChildren::Int64
	columnNumber::Int64
	columnType::Int64 # This should be an enumeratd data type
	switchValues::Array{Array{Number,1},1}
	childTreeIndex::Array{Int64}
	debug::Bool
	leafNode::Bool
	returnValue::Int64 # The class /regression value to be returned
	
	
	# Create a root
	function treeCell() 
		x = new()
		#------------------------------
		#	Initialize
		#------------------------------
		x.parentId = 0
		x.nodeId = 1
		x.numChildren = 0
		x.columnNumber = 0 # Impossible column
		x.columnType = 0 
		x.switchValues = Array[]
		x.childTreeIndex = Int64[]
		x.debug = false
		x.leafNode = true
		x.returnValue = 0
		return x
	end
end		# End type treeCell

type treeCellArray <:CART
	cellArray::Array{treeCell}
	#valArray::Array{Number}
	function treeCellArray()
		x = new()
		x.cellArray = treeCell[] #empty array
		z = treeCell() # Create a node
		push!(x.cellArray,z)
		return x
	end
end

function addNode!(root::treeCellArray, parentId::Int64, columnNumber::Int64, valueInColumn::Array)
	if parentId < 1
		error("parentId less than 1 is not permitted")
	end
	# First, create a new node
	x = treeCell()
	# Populate it
	x.parentId = parentId
	x.nodeId = length(root.cellArray) + 1
	x.numChildren = 0
	
	x.switchValues = Number[]
	x.childTreeIndex = Int64[]
	x.debug = false
	x.leafNode = true
	x.returnValue= 1
	
	push!(root.cellArray,x)
	#println("\t\t\tNumber of Nodes: ", length(root.cellArray))
	
	
	parent = root.cellArray[parentId]
	push!(parent.switchValues,vec(valueInColumn))
	
	parent.columnNumber = columnNumber
	push!(parent.childTreeIndex,x.nodeId)
	parent.numChildren += 1
	parent.leafNode = false
	parent.returnValue= 0
	return x.nodeId
end


function testtreeCellArray(root::treeCellArray, count::Int64)
	if count < 1
		error("The number of nodes to be generated (count) should be > 1")
	end
	# Create a tree of a given height
	# Root is already there
	parentId = 1 # Start with the root
	workQueue = Int64[]
	push!(workQueue,parentId)
	nodeCount = count
	
	while ((!isempty(workQueue)) && (nodeCount > 0))
		
		parentId = shift!(workQueue)
		numChildren = rand(2:5)
		for k = 1:numChildren
			if nodeCount == 1 # Remember, parent =1
				break
			end
			value = rand(100:150,3,1)
			#@printf("%s\n", typeof(value))
			
			childId = addNode!(root, parentId, 0, value)
			push!(workQueue, childId)
			nodeCount -= 1
		end
	end
	
	# Now traverse through the tree
	# Get the number of nodes
	for k = 1:count
		# println(k, " Generating tree ...")
		z = root.cellArray[k]
		# find the level
		# track back to parent
		level = 0
		ancestor = z
		while ancestor.parentId != 0
			ancestor = root.cellArray[ancestor.parentId]
			level += 1
		end
		println("\t" ^ level, z.nodeId)
	end
end

__currentTreeDepth__ = 0 # Special significance for the traverseTreeTest function

function traverseTreeTest(root::treeCellArray, nodeId::Int64)

	global __currentTreeDepth__
	
	if nodeId < 1
		error("nodeId cannot be less than 1")
	end
	
	if __currentTreeDepth__ < 0
		error("Number of levels in a tree cannot be negative")
	end
	
	
	z = root.cellArray[nodeId] # Get the tree cell
	println("\t" ^ __currentTreeDepth__, z.nodeId)
	if z.leafNode
		return
	end
	
	for k in z.childTreeIndex
		__currentTreeDepth__ += 1
		traverseTreeTest(root, k)
		__currentTreeDepth__ -= 1
	end
end
	


function getBestClass(y::Array{Int64}, uniqY::Array{Int64})
	maxCount = 0
	symbol = 0
	for k in uniqY
		count = length(find(z->z == k,y))
		maxCount = count > maxCount? count:maxCount
		symbol = count > maxCount? k:symbol
	end
	return symbol
end


function enumerateNode(root::treeCellArray, nodeIdSet::Array{Int64})
	
	numNodes = length(root.cellArray)
	
	for nodeId in nodeIdSet
		if (nodeId < 1) || (nodeId > numNodes)
			error("Invalid nodeId")
		else
			println(root.cellArray[nodeId])
		end
	end
end


function enumerateNode(root::treeCellArray)
	
	numNodes = length(root.cellArray)
	
	for nodeId = 1:numNodes
		println(root.cellArray[nodeId])
	end
	
end



	
function getReturnValues(root::treeCellArray)
	
	numNodes = length(root.cellArray)
	
	for nodeId = 1:numNodes
		z = root.cellArray[nodeId]
		if z.leafNode
			println("Node Id: ", z.nodeId, " Return Value: ", z.returnValue)
		end
	end
end
