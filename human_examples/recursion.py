#Return nth number in fibonacci sequence
def fib(n, a=0, b=1):
  if n <= 0:
    return b
  c = a + b
  a = b
  b = c
  return fib(n-1, a, b)

