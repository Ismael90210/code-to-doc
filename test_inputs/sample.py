def add(a, b):
    return a + b

def say_hello():
    print("Hello!")

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
