"""
Given the head of a singly linked list, reverse the list, and return the reversed list.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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
Given the head of a singly linked list, return true if it is a palindrome or false otherwise.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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
Given the head of a singly linked list and an integer k, split the linked list into k consecutive linked list parts.

The length of each part should be as equal as possible: no two parts should have a size differing by more than one. This may lead to some parts being null.

The parts should be in the order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal to parts occurring later.

Return an array of the k parts.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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
Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

"""
You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).
Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""
def swapNodes(self, head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    cur = head
    prev = None
    n = 0
    for i in range(k):
        if cur is None:
          return None
        prev = cur
        cur = cur.next
        n += 1
    lprev = prev
    l = cur
    while cur.next:
        n += 1
        cur = cur.next
    prev = None
    cur = head
    for i in range(n-k):
        prev = cur
        cur = cur.next
    r = cur
    rprev = prev

    tmplnext = l.next
    l.next = r.next
    r.next = tmplnext
    lprev.next = r
    rprev.next = l
    #Gone through k nodes