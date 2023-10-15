import json
p = [
{
"question":
"""Given the head of a singly linked list, reverse the list, and return the reversed list.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
""",
"code":
"""
def reverseList(head):
    if head == None:
        return None
    cur = head
    next = cur.next
    while next:
        tmp = next.next
        next.next = cur
        cur = next
        next = tmp

    head.next = None
    return cur
"""
},
{
"question":
"""Given the head of a singly linked list, return true if it is a palindrome or false otherwise.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
""",
"code" :
"""
def isPalindrome(head):
    slow = fast = head 
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next 
        
    stack1=[]
    stack2 = []
    
    fast = slow 
    slow = head
    while fast:
        stack2.append(slow.val)
        stack1.append(fast.val)
        fast = fast.next
        slow = slow.next 
        
    stack2.reverse()
    return stack1 == stack2
"""
},
{
"question":
"""Given the head of a singly linked list and an integer k, split the linked list into k consecutive linked list parts.

The length of each part should be as equal as possible: no two parts should have a size differing by more than one. This may lead to some parts being null.

The parts should be in the order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal to parts occurring later.

Return an array of the k parts.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
""",
"code" :
"""
def splitListToParts(head, k):
    cur = head
    N = 0
    while cur:
        N += 1
        cur = cur.next
    width, remainder = divmod(N, k)

    ans = []
    cur = head
    for i in range(k):
        head = cur
        for j in range(width + (i < remainder) - 1):
            if cur: cur = cur.next
        if cur:
            cur.next, cur = None, cur.next
        ans.append(head)
    return ans
"""
},
{
"question":
"""Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
""",
"code" :
"""
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
"""
},
{
"question":
"""You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
""",
"code" :
"""
def swapNodes(self, head, k):
    node1 = node2 = fast = head
    #Finding kth node from the start 
    k-=1
    while (k):
        node1 = node1.next
        fast = fast.next
        k-=1

    #Finding kth node from the end
    while (fast and fast.next):
        node2 = node2.next
        fast = fast.next

    #Swapping the values only
    temp = node1.val
    node1.val = node2.val
    node2.val = temp

    return head
"""
}
]
with open("linked_list_alg.json", 'w') as f:
  json.dump(p, f)