import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

def fun(x):
    return 2*(x) - 1

def fun_rand(x):
    return 2*(x)+random.random()/10-1


v_fun = np.vectorize(fun)
v_fun_rand = np.vectorize(fun_rand)

def random_mixture(x):
    _y = v_fun_rand(x)
    _x = x + np.random.rand(len(x))/10-.05
    return _x,_y


x = np.random.rand((100))
y1 = v_fun(x)
y2 = v_fun_rand(x)
x3, y3 = random_mixture(x)


fig = plt.Figure()
#plt.scatter(x,y1)
#plt.savefig("../data/out/simple_v1.png")
#plt.scatter(x,y2)
#plt.savefig("../data/out/simple_v2.png")
plt.scatter(x3,y3)
plt.savefig("../data/out/simple_v3.png")
plt.show()
plt.close()

_cov_matrix = np.cov(x,y1)
print(_cov_matrix)




data1 = np.column_stack((x,y1))
data1 = np.column_stack((x,y2))
data1 = np.column_stack((x3,y3))
##print(data1)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data1)
_com_matrix_pca = np.cov(pca_data[:,0], pca_data[:,1])
print(_com_matrix_pca)


print("pca.components_:")
print(pca.components_)
print("pca.explained_variance_:",pca.explained_variance_)
print("pca.explained_variance_ratio_:",pca.explained_variance_ratio_)



red_data = pca_data.copy()
red_data[:,1]=0
##print(red_data)
red_reversed = pca.inverse_transform(red_data)
fig = plt.Figure()
plt.scatter(x=data1[:,0], y=data1[:,1])
plt.scatter(x=red_reversed[:,0], y=red_reversed[:,1])
plt.show()
plt.close()

dy = data1[:,1]-red_reversed[:,1]
print("max diff", max(dy), "min diff:", min(dy))

exit(10)

# _cov_matrix = np.cov(data1[:,0],data1[:,1])
# print(_cov_matrix)
#
# _com_matrix_pca = np.cov(pca_data[:,0], pca_data[:,1])
# print(_com_matrix_pca)

