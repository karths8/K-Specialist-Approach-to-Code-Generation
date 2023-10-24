import json
p = [
{
"question": """Given an array reorder it so that even numbers appear first""",
"code": """
import numpy as np

def even_odd_reorder(arr: np.array):
  nextEven = 0
  nextOdd = len(arr) - 1
  while nextEven < nextOdd:
    if arr[nextEven] % 2 == 0:
      nextEven += 1
    else:
      temp = arr[nextEven]
      arr[nextEven] = arr[nextOdd]
      arr[nextOdd] = temp
      nextOdd -= 1
""",
"categories":["array, sorting"]
},
{
"question": """Given an array with 3 different values sort the array""",
"code": """
import numpy as np

def dutch_flag_partition(pivot_idx: int, arr: np.array):
  for i in range(len(arr)):
    for j in range(i + 1, len(arr)):
      if arr[j] < arr[pivot_idx]:
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
        break
  
  for i in range(len(arr) - 1, 0, -1):
    if arr[i] < arr[pivot_idx]:
      break
    for j in range(i - 1, 0, -1):
      if arr[j] < arr[pivot_idx]:
        break
      if arr[j] > arr[pivot_idx]:
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
        break
""",
"categories":["array, sorting"]
},
{
"question": """Given two arrays return an array with the set intersection of their common elements""",
"code": """
import numpy as np

def array_intersection(arr1: np.array, arr2: np.array):
  set1 = set(arr1)
  ret = [x for x in arr2 if x in set1]
  return np.array(ret)
""",
"categories":["array"]
},
{
"question": """Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.""",
"code": """
import numpy as np

def array_intersection_with_freq(arr1: np.array, arr2: np.array):
  freq = dict()
  ret = []
  for num in arr1:
      if num in freq.keys():
          freq[num] = freq[num]+1
      else:
          freq[num] = 1
  for num in arr2:
      if num in freq.keys() and freq[num] > 0:
          ret.append(num)
          freq[num] = freq[num] - 1
  return np.array(ret)
""",
"categories":["array"]
},
{
"question": """You are given two 0-indexed integer permutations A and B of length n.

A prefix common array of A and B is an array C such that C[i] is equal to the count of numbers that are present at or before the index i in both A and B.

Return the prefix common array of A and B.

A sequence of n integers is called a permutation if it contains all integers from 1 to n exactly once.""",
"code": """
import numpy as np

def findThePrefixCommonArray(arr1: np.array, arr2: np.array):
  max = arr1.size if arr1.size > arr2.size else arr2.size
  seen = set()
  curCount = 0
  ret = []
  for i in range(max):
    cur1 = arr1[i]
    cur2 = arr2[i]
    if cur1 in seen:
      curCount += 1
    else:
      seen.add(cur1)
    if cur2 in seen:
      curCount += 1
    else:
      seen.add(cur2)
    ret.append(curCount)
  return np.array(ret)
""",
"categories":["array"]
}
]
with open("human_examples/array_alg.json", 'w') as f:
  json.dump(p, f)