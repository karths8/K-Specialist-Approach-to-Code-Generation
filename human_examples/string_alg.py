
#https://leetcode.com/problems/reverse-string/
def reverse_string(s: str):
  l = 0
  r = len(s) - 1
  while l < r:
    tmp = s[l]
    s[l] = s[r]
    s[r] = tmp
    l += 1
    r -= 1

#https://leetcode.com/problems/permutation-in-string/editorial/
def permutation_in_string(s1: str, s2: str):
  window = len(s1)
  if window > len(s2):
    return False
  freq1 = [0]*26
  freq2 = [0]*26
  for i in range(window):
    freq1[ord(s1[i])-ord('a')] += 1
    freq2[ord(s2[i])-ord('a')] += 1

  matches = 0
  for i in range(26):
    if freq1[i] == freq2[i]:
      matches += 1
  
  for l in range(len(s2) - window):
    if matches == 26:
      return True

    added = ord(s2[l + window]) - ord('a')
    freq2[added] += 1
    if freq1[added] == freq2[added]:
      matches += 1
    elif freq1[added] + 1 == freq2[added]:
      matches -= 1

    removed = ord(s2[l]) - ord('a')
    freq2[removed] -= 1
    if freq1[removed] == freq2[removed]:
      matches += 1
    elif freq1[removed] == freq2[removed] + 1:
      matches -= 1
    
  return matches == 26

#https://leetcode.com/problems/shuffle-string/submissions/
def shuffle_string(s: str, indices: list[int]):
  ret = ['_']*len(s)
  for idx, i in enumerate(indices):
    ret[i] = s[idx]
  return "".join(ret)

#https://leetcode.com/problems/optimal-partition-of-string/description/
def partitionString(s):
  substrings = 1
  cur_str = set()
  for i in range(len(s)):
    if s[i] in cur_str:
      substrings += 1
      cur_str = {s[i]}
    else:
      cur_str.add(s[i])
  return substrings


#https://leetcode.com/problems/count-sorted-vowel-strings/description/
letters = ["a","e","i","o","u"]
dynamic = {
  'a1' : 5,
  'e1' : 4,
  'i1' : 3,
  'o1' : 2,
  'u1' : 1
}
def helper(n, letter_idx):
  ret = 0
  key = letters[letter_idx] + str(n)
  if key in dynamic.keys():
    return dynamic[key]
  else:
    for i in range(letter_idx, len(letters)):
      ret += helper(n-1, i)
    dynamic[key] = ret
  return ret
  

def countVowelStrings(n):
  """
  :type n: int
  :rtype: int
  """
  return helper(n, 0)

print(countVowelStrings(33))