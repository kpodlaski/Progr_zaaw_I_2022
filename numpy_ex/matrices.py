import numpy as np

def lin(a):
    return 2*a+3

vlin = np.vectorize(lin)

list = [[1,3,5],[7,19,2],[22,18,2]]
list2 = [[1,1,5],[7,10,2],[22,18,2]]

a1 = np.array(list)
a2 = np.array(list2)

print(a1)
print(a2)
print(a1*a1)
sd = (a1-a2)**2
print(sd)
print(sd.sum())
print(sd.sum(axis=0))
print(sd.sum(axis=1))
w = sd.sum(axis=1)
#print( w.reshape(1,1,3) )
#print( w.reshape(3,1,1, dtype=np.float64) )
print(np.matmul(a1,a2))
print(np.dot(a1,a2))

print(a1)
print(vlin(a1))