#Return nth number in the fibonacci sequence
def fib(n, a=0, b=1):
  if n <= 0:
    return b
  c = a + b
  a = b
  b = c
  return fib(n-1, a, b)

#Return the factorial of a given number n
def fact(n):
  if n <= 1:
    return 1
  return n*fact(n)

#Given a numerator and a denominator return the fraction in string format. If the fractional part is repeating enclose the repeating part in parentheses
def helper(cur, numerator, denominator, numerators_seen):
  if numerator == 0:
    return cur
  elif numerator in numerators_seen.keys():
    idx = numerators_seen[numerator]
    return cur[:idx] + '(' + cur [idx:] + ')'
  numerators_seen[numerator] = len(cur)
  if numerator >= denominator:
    cur += str(numerator // denominator)
    numerator = numerator % denominator
    return helper(cur, numerator*10, denominator, numerators_seen)
  else:
    cur += "0"
    return helper(cur, numerator*10, denominator, numerators_seen)

def fractionToDecimal(numerator, denominator):
  cur = ""
  if numerator == 0:
    return "0"
  elif (numerator < 0 and denominator > 0) or (numerator > 0 and denominator < 0):
    cur = "-"
  numerator = abs(numerator)
  denominator = abs(denominator)
  if numerator >= denominator:
    cur += str(numerator // denominator)
    numerator = numerator % denominator
    if numerator != 0:
      cur += "."
      return helper(cur, numerator*10, denominator, dict())
    else:
      return cur
  else:
    cur += "0."
    return helper(cur, numerator*10, denominator, dict())
  
"""You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.
Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, ight=None):
        self.val = val
        self.left = left
        self.right = right
"""
def forward(root, n):
    if root is None:
        return
    ret = forward(root.left, n)
    if ret:
        return ret
    
    if not n:
        n.append(root)
    elif root.val < n[0].val:
        return n[0]
    else:
        n[0] = root
    
    ret = forward(root.right, n)
    if ret:
        return ret
    
def backward(root, n):
    if root is None:
        return
    
    ret = backward(root.right, n)
    if ret:
        return ret
    
    if not n:
        n.append(root)
    elif root.val > n[0].val:
        return n[0]
    else:
        n[0] = root
    
    ret = backward(root.left, n)
    if ret:
        return ret
    
def recoverTree(root):
    x, y = forward(root, []), backward(root, [])
    x.val, y.val = y.val, x.val
    return root


"""Given a list of values output a root binary tree node with appropiate children. None should not have a Tree Node created for them.
Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, ight=None):
        self.val = val
        self.left = left
        self.right = right
"""
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def createTree(nums):
   if len(nums) == 0:
      return None
   def helper(cur):
      if cur >= len(nums):
         return None
      left = helper(cur*2+1)
      right = helper(cur*2+2)
      if nums[cur] is not None:
        return TreeNode(nums[cur], left, right)
      else:
         return None
   return helper(0)