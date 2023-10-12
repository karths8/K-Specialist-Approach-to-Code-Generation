#https://www.geeksforgeeks.org/0-1-knapsack-using-branch-and-bound/
"""
Given two integer arrays val[0..n-1] and wt[0..n-1] that represent values and weights associated with n items respectively. 

Find out the maximum value subset of val[] such that sum of the weights of this subset is smaller than or equal to Knapsack capacity W.
"""

from queue import Queue
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value
class Node:
    def __init__(self, level, profit, bound, weight):
        self.level = level
        self.profit = profit
        self.bound = bound
        self.weight = weight
def compare(a, b):
    r1 = float(a.value) / a.weight
    r2 = float(b.value) / b.weight
    return r1 > r2
 
def bound(u, n, W, arr):
    if u.weight >= W:
        return 0
    profitBound = u.profit
    j = u.level + 1
    totWeight = int(u.weight)
 
    while j < n and totWeight + int(arr[j].weight) <= W:
        totWeight += int(arr[j].weight)
        profitBound += arr[j].value
        j += 1

    if j < n:
        profitBound += int((W - totWeight) * arr[j].value / arr[j].weight)
 
    return profitBound
 
def knapsack_solution(W, arr, n):
    arr.sort(cmp=compare, reverse=True)

    q = Queue()
    u = Node(-1, 0, 0, 0)
    q.put(u)

    maxProfit = 0
 
    while not q.empty():
        u = q.get()
 
        if u.level == -1:
            v = Node(0, 0, 0, 0)

        if u.level == n - 1:
            continue

        v = Node(u.level + 1, u.profit +
                 arr[u.level + 1].value, 0, u.weight + arr[u.level + 1].weight)

        if v.weight <= W and v.profit > maxProfit:
            maxProfit = v.profit
 
        v.bound = bound(v, n, W, arr)
 
        if v.bound > maxProfit:
            q.put(v)
 
        v = Node(u.level + 1, u.profit, 0, u.weight)
 
        v.bound = bound(v, n, W, arr)
 
        if v.bound > maxProfit:
            q.put(v)
 
    return maxProfit

"""
Given a 3×3 board with 8 tiles (every tile has one number from 1 to 8) and one empty space. The objective is to place the numbers on tiles to match the final configuration using the empty space. We can slide four adjacent (left, right, above, and below) tiles into the empty space. 

For example, 
initial: [[1,2,3],[5,6,None],[7,8,4]]
final: [[1,2,3],[5,8,6],[None,7,4]]
"""
import copy
from heapq import heappush, heappop
 
# This variable can be changed to change
# the program from 8 puzzle(n=3) to 15 
# puzzle(n=4) to 24 puzzle(n=5)...
n = 3
 
# bottom, left, top, right
row = [ 1, 0, -1, 0 ]
col = [ 0, -1, 0, 1 ]
 
# A class for Priority Queue
class priorityQueue:
     
    # Constructor to initialize a
    # Priority Queue
    def __init__(self):
        self.heap = []
 
    # Inserts a new key 'k'
    def push(self, k):
        heappush(self.heap, k)
 
    # Method to remove minimum element 
    # from Priority Queue
    def pop(self):
        return heappop(self.heap)
 
    # Method to know if the Queue is empty
    def empty(self):
        if not self.heap:
            return True
        else:
            return False
 
# Node structure
class node:
     
    def __init__(self, parent, mat, empty_tile_pos,
                 cost, level):
                      
        # Stores the parent node of the 
        # current node helps in tracing 
        # path when the answer is found
        self.parent = parent
 
        # Stores the matrix
        self.mat = mat
 
        # Stores the position at which the
        # empty space tile exists in the matrix
        self.empty_tile_pos = empty_tile_pos
 
        # Stores the number of misplaced tiles
        self.cost = cost
 
        # Stores the number of moves so far
        self.level = level
 
    # This method is defined so that the 
    # priority queue is formed based on 
    # the cost variable of the objects
    def __lt__(self, nxt):
        return self.cost < nxt.cost
 
# Function to calculate the number of 
# misplaced tiles ie. number of non-blank
# tiles not in their goal position
def calculateCost(mat, final) -> int:
     
    count = 0
    for i in range(n):
        for j in range(n):
            if ((mat[i][j]) and
                (mat[i][j] != final[i][j])):
                count += 1
                 
    return count
 
def newNode(mat, empty_tile_pos, new_empty_tile_pos,
            level, parent, final) -> node:
                 
    # Copy data from parent matrix to current matrix
    new_mat = copy.deepcopy(mat)
 
    # Move tile by 1 position
    x1 = empty_tile_pos[0]
    y1 = empty_tile_pos[1]
    x2 = new_empty_tile_pos[0]
    y2 = new_empty_tile_pos[1]
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]
 
    # Set number of misplaced tiles
    cost = calculateCost(new_mat, final)
 
    new_node = node(parent, new_mat, new_empty_tile_pos,
                    cost, level)
    return new_node
 
# Function to print the N x N matrix
def printMatrix(mat):
     
    for i in range(n):
        for j in range(n):
            print("%d " % (mat[i][j]), end = " ")
             
        print()
 
# Function to check if (x, y) is a valid
# matrix coordinate
def isSafe(x, y):
     
    return x >= 0 and x < n and y >= 0 and y < n
 
# Print path from root node to destination node
def printPath(root):
     
    if root == None:
        return
     
    printPath(root.parent)
    printMatrix(root.mat)
    print()
 
# Function to solve N*N - 1 puzzle algorithm
# using Branch and Bound. empty_tile_pos is
# the blank tile position in the initial state.
def solve(initial, empty_tile_pos, final):
     
    # Create a priority queue to store live
    # nodes of search tree
    pq = priorityQueue()
 
    # Create the root node
    cost = calculateCost(initial, final)
    root = node(None, initial, 
                empty_tile_pos, cost, 0)
 
    # Add root to list of live nodes
    pq.push(root)
 
    # Finds a live node with least cost,
    # add its children to list of live 
    # nodes and finally deletes it from 
    # the list.
    while not pq.empty():
 
        # Find a live node with least estimated
        # cost and delete it from the list of 
        # live nodes
        minimum = pq.pop()
 
        # If minimum is the answer node
        if minimum.cost == 0:
             
            # Print the path from root to
            # destination;
            printPath(minimum)
            return
 
        # Generate all possible children
        for i in range(4):
            new_tile_pos = [
                minimum.empty_tile_pos[0] + row[i],
                minimum.empty_tile_pos[1] + col[i], ]
                 
            if isSafe(new_tile_pos[0], new_tile_pos[1]):
                 
                # Create a child node
                child = newNode(minimum.mat,
                                minimum.empty_tile_pos,
                                new_tile_pos,
                                minimum.level + 1,
                                minimum, final,)
 
                # Add child to list of live nodes
                pq.push(child)

"""
Given an input N for a NxN chessboard place N queens such that no two queens attack each other. Return the NxN board with the Queens marked as Q. If the problem is not solvable return Solution does not exist
"""
N = 8

def printSolution(board):
    for i in range(N):
        for j in range(N):
            print(board[i][j], end = " ")
        print()

def isSafe(row, col, slashCode, backslashCode, 
           rowLookup, slashCodeLookup, 
                       backslashCodeLookup):
    if (slashCodeLookup[slashCode[row][col]] or
        backslashCodeLookup[backslashCode[row][col]] or
        rowLookup[row]):
        return False
    return True

def solveNQueensUtil(board, col, slashCode, backslashCode, 
                     rowLookup, slashCodeLookup,
                     backslashCodeLookup):
    if(col >= N):
        return True
    for i in range(N):
        if(isSafe(i, col, slashCode, backslashCode, 
                  rowLookup, slashCodeLookup,
                  backslashCodeLookup)):
            board[i][col] = 1
            rowLookup[i] = True
            slashCodeLookup[slashCode[i][col]] = True
            backslashCodeLookup[backslashCode[i][col]] = True
            if(solveNQueensUtil(board, col + 1, 
                                slashCode, backslashCode, 
                                rowLookup, slashCodeLookup, 
                                backslashCodeLookup)):
                return True
            board[i][col] = 0
            rowLookup[i] = False
            slashCodeLookup[slashCode[i][col]] = False
            backslashCodeLookup[backslashCode[i][col]] = False
             
    return False
def solveNQueens():
    board = [[0 for i in range(N)] 
                for j in range(N)]
     
    # helper matrices 
    slashCode = [[0 for i in range(N)] 
                    for j in range(N)]
    backslashCode = [[0 for i in range(N)] 
                        for j in range(N)]
     
    # arrays to tell us which rows are occupied 
    rowLookup = [False] * N
     
    # keep two arrays to tell us 
    # which diagonals are occupied 
    x = 2 * N - 1
    slashCodeLookup = [False] * x
    backslashCodeLookup = [False] * x
     
    # initialize helper matrices 
    for rr in range(N):
        for cc in range(N):
            slashCode[rr][cc] = rr + cc
            backslashCode[rr][cc] = rr - cc + 7
     
    if(solveNQueensUtil(board, 0, slashCode, backslashCode, 
                        rowLookup, slashCodeLookup, 
                        backslashCodeLookup) == False):
        print("Solution does not exist")
        return False
         
    # solution found 
    printSolution(board)
    return True

"""
Let there be N workers and N jobs. Any worker can be assigned to perform any job, incurring some cost that may vary depending on the work-job assignment. It is required to perform all jobs by assigning exactly one worker to each job and exactly one job to each agent in such a way that the total cost of the assignment is minimized.
Your input will be an NxN matrix of cost for each work to do each job. Return an assignment matrix with the least cost.
"""
import math
from heapq import heappush, heappop

class priorityQueue:
     
    # Constructor to initialize a
    # Priority Queue
    def __init__(self):
        self.heap = []
 
    # Inserts a new key 'k'
    def push(self, k):
        heappush(self.heap, k)
 
    # Method to remove minimum element 
    # from Priority Queue
    def pop(self):
        return heappop(self.heap)
 
    # Method to know if the Queue is empty
    def empty(self):
        if not self.heap:
            return True
        else:
            return False

class Node():
    def __init__(self, parent, pathCost, cost, workerID, jobID, assigned):
        self.parent = parent
        self.pathCost = pathCost
        self.cost = cost
        self.workerID = workerID
        self.jobId = jobID
        self.assigned = assigned

    def __lt__(self, other):
        return self.cost < other.cost

def newNode(x, y, assigned, parent):
    a = copy(assigned)
    a[y] = True
    return Node(parent, None, None, x, y, a)

def calcCost(cost_matrix, x, y, assigned):
    cost = 0
    available = [True] * len(cost_matrix)
    for i in range(x+1, len(cost_matrix)):
        min = math.inf
        minIndex = -1
        for j in range(len(cost_matrix)):
            if not assigned[j] and available[j] and cost_matrix[i][j] < min:
                minIndex = j
                min = cost_matrix[i][j]
        cost += min
        available[minIndex] = False

    return cost

def retAssignment(min: Node, n):
    ret = []
    for i in range(n):
        ret.append([False]*n)
    cur = min
    while cur is not None:
        ret[cur.workerID][cur.jobId] = True
        cur = cur.parent

def find_min_cost(cost_matrix):
    pq = priorityQueue()
    assigned = [False] * len(cost_matrix)
    root = newNode(-1,-1, assigned, None)
    root.pathCost = 0
    root.cost = 0
    root.workerID = -1
    pq.push(root)
    while not pq.empty():
        min = pq.pop()
        i = min.workerID + 1
        if i == len(cost_matrix):
            return retAssignment(min, len(cost_matrix))
        for j in range(len(cost_matrix)):
            child = newNode(i,j,min.assigned, min)
            child.pathCost = min.pathCost + cost_matrix[i][j]
            child.cost = child.pathCost + calcCost(cost_matrix, i, j, child.assigned)
            pq.push(child)

"""
Given a set of cities and distance between every pair of cities, the problem is to find the shortest possible tour that visits every city exactly once and returns to the starting point.

"""
import math
maxsize = float('inf')
def copyToFinal(curr_path):
	final_path[:N + 1] = curr_path[:]
	final_path[N] = curr_path[0]
def firstMin(adj, i):
	min = maxsize
	for k in range(N):
		if adj[i][k] < min and i != k:
			min = adj[i][k]

	return min

def secondMin(adj, i):
	first, second = maxsize, maxsize
	for j in range(N):
		if i == j:
			continue
		if adj[i][j] <= first:
			second = first
			first = adj[i][j]

		elif(adj[i][j] <= second and
			adj[i][j] != first):
			second = adj[i][j]

	return second

def TSPRec(adj, curr_bound, curr_weight, 
			level, curr_path, visited):
	global final_res

	if level == N:

		if adj[curr_path[level - 1]][curr_path[0]] != 0:
			curr_res = curr_weight + adj[curr_path[level - 1]]\
										[curr_path[0]]
			if curr_res < final_res:
				copyToFinal(curr_path)
				final_res = curr_res
		return

	for i in range(N):
		if (adj[curr_path[level-1]][i] != 0 and
							visited[i] == False):
			temp = curr_bound
			curr_weight += adj[curr_path[level - 1]][i]
			if level == 1:
				curr_bound -= ((firstMin(adj, curr_path[level - 1]) +
								firstMin(adj, i)) / 2)
			else:
				curr_bound -= ((secondMin(adj, curr_path[level - 1]) +
								firstMin(adj, i)) / 2)
			if curr_bound + curr_weight < final_res:
				curr_path[level] = i
				visited[i] = True
				TSPRec(adj, curr_bound, curr_weight, 
					level + 1, curr_path, visited)
			curr_weight -= adj[curr_path[level - 1]][i]
			curr_bound = temp
			visited = [False] * len(visited)
			for j in range(level):
				if curr_path[j] != -1:
					visited[curr_path[j]] = True
def TSP(adj):
	curr_bound = 0
	curr_path = [-1] * (N + 1)
	visited = [False] * N
	for i in range(N):
		curr_bound += (firstMin(adj, i) +
					secondMin(adj, i))
	curr_bound = math.ceil(curr_bound / 2)
	visited[0] = True
	curr_path[0] = 0
	TSPRec(adj, curr_bound, 0, 1, curr_path, visited)

# Driver code

# Adjacency matrix for the given graph
adj = [[0, 10, 15, 20],
	[10, 0, 35, 25],
	[15, 35, 0, 30],
	[20, 25, 30, 0]]
N = 4

# final_path[] stores the final solution 
# i.e. the // path of the salesman.
final_path = [None] * (N + 1)

# visited[] keeps track of the already
# visited nodes in a particular path
visited = [False] * N

# Stores the final minimum weight
# of shortest tour.
final_res = maxsize

TSP(adj)

print("Minimum cost :", final_res)
print("Path Taken : ", end = ' ')
for i in range(N + 1):
	print(final_path[i], end = ' ')
