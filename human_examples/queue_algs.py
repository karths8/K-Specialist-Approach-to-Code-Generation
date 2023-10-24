import json
p = [
{
"question":
""" Given a queue reverse it
""",
"code" :
"""
from collections import deque
def reverse(queue: deque):
    if not len(queue) == 0:
        temp = queue.pop()
        reverse(queue)
        queue.appendleft(temp)
""",
"categories":["queue"]
},
{
"question":
"""Given a root treenode convert the entire tree into a list of values such that the nth entry has children 2*n+1 and 2*n+2.
class TreeNode(object):
    def __init__(self, val=0, left=None, ight=None):
        self.val = val
        self.left = left
        self.right = right
""",
"code" :
"""
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_to_list(root: TreeNode):
    ret = []
    if root is None:
        return ret
    cur_queue = [root]
    still_nodes = True
    while still_nodes:
        still_nodes = False
        next_queue = []
        for node in cur_queue:
            left = None
            right = None
            if node is not None:
                left = node.left
                right = node.right
                ret.append(node.val)
            else:
                ret.append(None)
            next_queue.append(left)
            next_queue.append(right)
            if left is not None or right is not None:
                still_nodes = True
        cur_queue = next_queue
    return ret
""",
"categories":["queue", "tree"]
},
{
"question":
"""Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO (First In First Out) principle, and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

Implement the MyCircularQueue class:

MyCircularQueue(k) Initializes the object with the size of the queue to be k.
int Front() Gets the front item from the queue. If the queue is empty, return -1.
int Rear() Gets the last item from the queue. If the queue is empty, return -1.
boolean enQueue(int value) Inserts an element into the circular queue. Return true if the operation is successful.
boolean deQueue() Deletes an element from the circular queue. Return true if the operation is successful.
boolean isEmpty() Checks whether the circular queue is empty or not.
boolean isFull() Checks whether the circular queue is full or not.
""",
"code" :
"""
class MyCircularQueue:
    def __init__(self, k):
        self.size=k
        self.q=[0]*k
        self.front=-1
        self.rear=-1

    def enQueue(self, value):
        if self.isFull():
            return False
        if self.isEmpty():
            self.front=self.rear=0
        else:
            self.rear=(self.rear+1)%self.size
        self.q[self.rear]=value
        return True

    def deQueue(self):
        if self.isEmpty():
            return False
        item=self.q[self.front]
        if self.front==self.rear:
            self.front=self.rear=-1
        else:
            self.front=(self.front+1)%self.size
        return True

    def Front(self):
        if self.front==-1:
            return -1
        return self.q[self.front]

    def Rear(self):
        if self.rear==-1:
            return -1
        return self.q[self.rear]

    def isEmpty(self):
        return self.front==-1

    def isFull(self):
        if self.front==0 and self.rear==self.size-1:
            return True
        if self.rear==(self.front-1)%(self.size):
            return True
        return False
""",
"categories":["queue"]
},
{
"question":
"""
Design a queue that supports push and pop operations in the front, middle, and back.

Implement the FrontMiddleBack class:

FrontMiddleBack() Initializes the queue.
void pushFront(int val) Adds val to the front of the queue.
void pushMiddle(int val) Adds val to the middle of the queue.
void pushBack(int val) Adds val to the back of the queue.
int popFront() Removes the front element of the queue and returns it. If the queue is empty, return -1.
int popMiddle() Removes the middle element of the queue and returns it. If the queue is empty, return -1.
int popBack() Removes the back element of the queue and returns it. If the queue is empty, return -1.
Notice that when there are two middle position choices, the operation is performed on the frontmost middle position choice
""",
"code" :
"""
class FrontMiddleBackQueue(object):
    def __init__(self):
        self.queue=[]

    def pushFront(self, val):
        self.queue.insert(0,val)

    def pushMiddle(self, val):
        mid=(len(self.queue))//2
        self.queue.insert(mid,val)
    
    def pushBack(self, val):
        self.queue.append(val)

    def popFront(self):
        if len(self.queue)==0:
            return -1
        else:
            res=self.queue.pop(0)
            return res
    
    def popMiddle(self):
        if len(self.queue)==0:
            return -1
        else:
            mid=(len(self.queue)-1)//2
            res=self.queue.pop(mid)
            return res
    
    def popBack(self):
        if len(self.queue)==0:
            return -1
        else:
            res=self.queue.pop()
            return res
""",
"categories":["queue"]
},
{
"question":  
"""You are given a string s and an integer k. You can choose one of the first k letters of s and append it at the end of the string.

Return the lexicographically smallest string you could have after applying the mentioned step any number of moves.
""",
"code" :
"""
def orderlyQueue(s: str, k: int) -> str:
    if k == 1:
        n = len(s)
        ss = s + s
        lex_min_str = s
        for i in range(n):
            lex_min_str = min(lex_min_str, ss[i:i + n])
        return lex_min_str
    
    return "".join(sorted(s))
""",
"categories":["queue", "string"]
}
]
with open("human_examples/queue_algs.json", 'w') as f:
  json.dump(p, f)