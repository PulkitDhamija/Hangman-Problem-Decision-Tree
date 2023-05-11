import numpy as np

def my_fit( words ):
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	dt.fit( words )
	return dt


class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words):
		self.words = words
		self.root = Node( depth = 0)
		# The root is trained with all the words
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth)


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth):
		self.depth = depth
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query( self ):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			response = ''.join(response.split(' '))
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	def reveal( self, word, query ):
		# Find out the intersections between the query and the word
		mask = ['_']*len(word)
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ''.join( mask )
	
	# Dummy node splitting action -- use a random word as query
	# Note that any word in the dictionary can be the query
	def process_node( self, all_words, my_words_idx):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		split_dict = {}
		query_idx = -1
		
		if self.depth == 0:
			query = ""
			for idx in my_words_idx:
				mask = self.reveal( all_words[ idx ], query )
				if mask not in split_dict:
					split_dict[ mask ] = []
                
				split_dict[ mask ].append( idx )
		else:
			entropy = -np.Inf
			my_words_len = len(my_words_idx)
			for curr_query_idx in my_words_idx:
				curr_query = all_words[ curr_query_idx ]
				curr_split_dict = {}

				for idx in my_words_idx:
					mask = self.reveal( all_words[ idx ], curr_query )
					if mask not in curr_split_dict:
						curr_split_dict[ mask ] = []
					curr_split_dict[ mask ].append( idx )

				curr_entropy = 0
				for j in map(len, curr_split_dict.values()):
					k = j/my_words_len
					curr_entropy -= k*np.log2(k)

				if curr_entropy > entropy:
					entropy = curr_entropy
					query_idx = curr_query_idx
					split_dict = curr_split_dict

		return ( query_idx, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = my_words_idx[0]
		else:
			self.is_leaf = False
			self.query_idx, split_dict = self.process_node( self.all_words, self.my_words_idx)
			
			for response, split in split_dict.items():				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1)
				
				# Recursively train this child node
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth)
