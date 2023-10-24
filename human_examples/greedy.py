import json
p = [
{
"question":
"""You are given an array of people, people, which are the attributes of some people in a queue (not necessarily in order). Each people[i] = [hi, ki] represents the ith person of height hi with exactly ki other people in front who have a height greater than or equal to hi.

Reconstruct and return the queue that is represented by the input array people. The returned queue should be formatted as an array queue, where queue[j] = [hj, kj] is the attributes of the jth person in the queue (queue[0] is the person at the front of the queue).
""",
"code":
"""
def reconstructQueue(people):
    output=[] 
    people.sort(key=lambda x: (-x[0], x[1]))                
    for a in people:
        output.insert(a[1], a)
    
    return output
""",
"categories":["greedy", "array"]
},
{
"question":
"""There are N Mice and N holes are placed in a straight line. Each hole can accommodate only 1 mouse. A mouse can stay at his position, move one step right from x to x + 1, or move one step left from x to x -1. Any of these moves consumes 1 minute. Assign mice to holes so that the time when the last mouse gets inside a hole is minimized.
You are given a list of mice and hole positions as integer scalers.
""",
"code":
"""
def assignHole(mices, holes):
    if (len(mices) != len(holes)):
        return -1
    mices.sort()
    holes.sort()
    Max = 0
     
    for i in range(len(holes)):
        if (Max < abs(mices[i] - holes[i])):
            Max = abs(mices[i] - holes[i])
     
    return Max
""",
"categories":["greedy", "array", "sorting"]
},
{
"question":
"""Every positive fraction can be represented as sum of unique unit fractions. A fraction is unit fraction if numerator is 1 and denominator is a positive integer, for example 1/3 is a unit fraction. Such a representation is called Egyptian Fraction.
Given a numerator and denominator print the  Egyptian Fraction.
""",
"code":
"""
import math
def egyptianFraction(nr, dr):
 
    print("The Egyptian Fraction " +
          "Representation of {0}/{1} is".
                format(nr, dr), end="\n")
    ef = []
    while nr != 0:
        x = math.ceil(dr / nr)
        ef.append(x)
        nr = x * nr - dr
        dr = dr * x
    for i in range(len(ef)):
        if i != len(ef) - 1:
            print(" 1/{0} +" . 
                    format(ef[i]), end = " ")
        else:
            print(" 1/{0}" .
                    format(ef[i]), end = " ")
""",
"categories":["greedy", "mathematical"]
},
{
"question":         
"""Every house in the colony has at most one pipe going into it and at most one pipe going out of it. Tanks and taps are to be installed in a manner such that every house with one outgoing pipe but no incoming pipe gets a tank installed on its roof and every house with only an incoming pipe and no outgoing pipe gets a tap.

Given two integers n and p denoting the number of houses and the number of pipes. The connections of pipe among the houses contain three input values: a_i, b_i, d_i denoting the pipe of diameter d_i from house a_i to house b_i, find out the efficient solution for the network. 

The output will contain the number of pairs of tanks and taps t installed in first line and the next t lines contain three integers: house number of tank, house number of tap and the minimum diameter of pipe between them.

Example:
Input:  4 2
        1 2 60
        3 4 50
Output: 2
        1 2 60
        3 4 50
""",
"code":
"""
def dfs(w, ans, cd, wt):
	if (cd[w] == 0):
		return w, ans
	if (wt[w] < ans):
		ans = wt[w]
	return dfs(cd[w], ans, cd, wt)

# Function performing calculations.
def solve(n, p, arr):
    # Array rd stores the 
    # ending vertex of pipe
    rd = [0]*1100

    # Array wd stores the value 
    # of diameters between two pipes
    wt = [0]*1100

    # Array cd stores the 
    # starting end of pipe
    cd = [0]*1100
    ans = 0
    i = 0
    while (i < p):
        q = arr[i][0]
        h = arr[i][1]
        t = arr[i][2]
        
        cd[q] = h
        wt[q] = t
        rd[h] = q
        i += 1
    a = []
    b = []
    c = []
	
    for j in range(1, n + 1):
        if (rd[j] == 0 and cd[j]):
            
            ans = 1000000000
            w, ans = dfs(j, ans, cd, wt)
            
            # We put the details of component
            # in final output array
            a.append(j)
            b.append(w) 
            c.append(ans)
    print(len(a))
    for j in range(len(a)):
        print(a[j], b[j], c[j])
""",
"categories":["greedy", "graph"]
},
{
"question":
"""You are given n activities with their start and finish times. Select the maximum number of activities that can be performed by a single person and return their indexes. Assume that a person can only work on a single activity at a time. 
Example:
Input: start[]  =  {10, 12, 20}, finish[] =  {20, 25, 30}
Output: 0 2
""",
"code":
"""
import numpy as np
def printMaxActivities(s, f):
    indices = np.argsort(f)
    [s[i] for i in indices]
    [f[i] for i in indices]
    n = len(f)
    print("Following activities are selected")
    i = 0
    print(i, end=' ')

    # Consider rest of the activities
    for j in range(1, n):

        # If this activity has start time greater than
        # or equal to the finish time of previously
        # selected activity, then select it
        if s[j] >= f[i]:
            print(j, end=' ')
            i = j
""",
"categories":["greedy"]
}
]
with open("human_examples/greedy.json", 'w') as f:
  json.dump(p, f)