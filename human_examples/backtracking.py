import json
p = [
{
"question": """Given a N*N board with the Knight placed on the first block of an empty board. Moving according to the rules of chess knight must visit each square exactly once. Print the order of each cell in which they are visited.
""",
"code":
"""
def isSafe(x, y, board, n): 
  if(x >= 0 and y >= 0 and x < n and y < n and board[x][y] == -1): 
    return True
  return False

def printSolution(n, board):
  for i in range(n): 
    for j in range(n): 
      print(board[i][j], end=' ') 
    print() 
  
  
def solveKT(n): 
  board = [[-1 for i in range(n)]for i in range(n)]
  move_x = [2, 1, -1, -2, -2, -1, 1, 2] 
  move_y = [1, 2, 2, 1, -1, -2, -2, -1] 
  board[0][0] = 0 
  pos = 1
  if(not solveKTUtil(n, board, 0, 0, move_x, move_y, pos)): 
      print("Solution does not exist") 
  else: 
      printSolution(n, board) 
  
  
def solveKTUtil(n, board, curr_x, curr_y, move_x, move_y, pos): 
  if(pos == n**2): 
    return True
  for i in range(8): 
    new_x = curr_x + move_x[i] 
    new_y = curr_y + move_y[i] 
    if(isSafe(new_x, new_y, board, n)): 
      board[new_x][new_y] = pos 
      if(solveKTUtil(n, board, new_x, new_y, move_x, move_y, pos+1)): 
        return True
      board[new_x][new_y] = -1
  return False
"""
},
{
"question": 
"""A Maze is given as N*N binary matrix of blocks where source block is maze[0][0] and destination block is maze[N-1][N-1]. 
A rat starts from source and has to reach the destination. The rat can move only in two directions: forward and down. 

In the maze matrix, 0 means the block is a dead end and 1 means the block can be used in the path from source to destination. 
Given an N*N input maze Output a N*N matrix where all entries in the solution path are marked as 1 or output Solution does not exist if no solution exists.
""",
"code" :
"""
def isValid(n, maze, x, y, res):
  if x >= 0 and y >= 0 and x < n and y < n and maze[x][y] == 1 and res[x][y] == 0:
    return True
  return False

def RatMaze(n, maze, move_x, move_y, x, y, res):
  if x == n-1 and y == n-1:
    return True
  for i in range(4):
    x_new = x + move_x[i]
    y_new = y + move_y[i]
    if isValid(n, maze, x_new, y_new, res):
      res[x_new][y_new] = 1
      if RatMaze(n, maze, move_x, move_y, x_new, y_new, res):
          return True
      res[x_new][y_new] = 0
  return False
 
def solveMaze(maze):
    n = len(maze)
    res = [[0 for i in range(n)] for i in range(n)]
    res[0][0] = 1
    move_x = [-1, 1, 0, 0]
    move_y = [0, 0, -1, 1]
    if RatMaze(n, maze, move_x, move_y, 0, 0, res):
        for i in range(n):
            for j in range(n):
                print(res[i][j], end=' ')
            print()
    else:
        print('Solution does not exist')
"""
},
{
"question": """Given an input N for a NxN chessboard place N queens such that no two queens attack each other. Return the NxN board with the Queens marked as Q. If the problem is not solvable return Solution does not exist
""",
"code":
"""
def printSolution(board):
  for i in range(len(board)):
    for j in range(len(board)):
      if board[i][j] == 1:
        print("Q",end=" ")
      else:
        print(".",end=" ")
    print()

def isSafe(board, row, col):
  for i in range(col):
    if board[row][i] == 1:
      return False
  for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
    if board[i][j] == 1:
      return False
  for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
    if board[i][j] == 1:
      return False

  return True
 
def solveNQUtil(board, col):
  if col >= len(board):
    return True
  for i in range(len(board)):
    if isSafe(board, i, col):
      board[i][col] = 1
      if solveNQUtil(board, col + 1) == True:
        return True
      board[i][col] = 0
  return False

def solveNQ(N):
  board = []
  for row in range(N):
    board.append([0]*N)
  if solveNQUtil(board, 0) == False:
    print("Solution does not exist")
  printSolution(board)
"""
},
{
"question":
"""Given a set[] of non-negative integers and a value sum, return the subset of the given set whose sum is equal to the given sum or None if one doesn't exist
""",
"code":
"""
def subsetSum_helper(cur_idx: int, fullset: list, target_sum: int, subset: list):
  if target_sum == 0:
    return subset

  if (cur_idx == len(fullset)):
    return None
  
  excluding = subsetSum_helper(cur_idx + 1, fullset, target_sum, subset)
  if excluding is not None:
    return excluding
  elif fullset[cur_idx] <= target_sum:
    subset.append(fullset[cur_idx])
    including = subsetSum_helper(cur_idx + 1, fullset, target_sum - fullset[cur_idx], subset)
    if including is not None:
      return including
    else:
      subset.pop()
  return None

def subsetsum(fullset: list, target_sum: int):
  return subsetSum_helper(0, fullset, target_sum, [])
"""
},
{
"question": "Given an undirected graph in the form of an adjacency matrix and a number m color the graph with m colors such that no two adjacent vertices share the same color. Return a list of colors for each vertex or None if no solution exists.",
"code":
"""
def isSafe(graph, v, colour, c):
  for i in range(len(graph)):
    if graph[v][i] == 1 and colour[i] == c:
      return False
  return True

def graphColourUtil(graph, m, colour, v):
  if v == len(graph):
    return True

  for c in range(1, m + 1):
    if isSafe(graph, v, colour, c) == True:
      colour[v] = c
      if graphColourUtil(graph, m, colour, v + 1) == True:
        return True
      colour[v] = 0

def graphColouring(graph:[[int]], m):
  colour = [0] * len(graph)
  if graphColourUtil(graph, m, colour, 0):
    return colour
  else:
    return None
"""
}
]
print(p)
with open("backtracking.json", 'w') as f:
  json.dump(p, f)