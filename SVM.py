import numpy as np
#Multiclass SVM Loss
def L_i_vectorized(x,y,W):
    scores = W.dot(x)
    margins = np.maximum(0,scores - scores[y]+1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


cat = [3.2,5.1,-1.7]
car = [1.3,4.9,2.5]
frog = [2.2,2.5,-3.1]

labels = [cat,car,frog]


loss=0
for i in range(len(labels)):
    Li = 0
    for j in labels[i]:
        if j!=labels[i][i]:
            Li += max(0,j-labels[i][i]+1)

    loss+=Li

print(round(loss/len(labels),2))


import math

a = math.exp(3.1)
b = math.exp(3.2)
c = math.exp(3.1)

S = a+b+c
print('cat = ',round(a/S,2)*100,'%')
print('car = ',round(b/S,2)*100,'%')
print('frog = ',round(c/S,2)*100,'%')


print(-math.log(round(a/S,2)))
print(-math.log(round(b/S,2)))




