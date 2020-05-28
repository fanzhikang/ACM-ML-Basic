import numpy as np

import matplotlib.pyplot as plt  

file = "./data/datasets_374192_727480_data.csv"

data = np.loadtxt(file,delimiter=',')
x_total = data[:,1:3]
y_total = data[:,3]

pos_index = np.where(y_total==1)
neg_index = np.where(y_total==0)

#plt.scatter(x_total[pos_index,0],x_total[pos_index,1],marker='o',c='r')
#plt.scatter(x_total[neg_index,0],x_total[neg_index,1],marker='x',c='b')
#plt.show()

from sklearn import linear_model

model = linear_model.LogisticRegression()
model.fit(x_total,y_total)

y_pred = model.predict(x_total)
print("accuracy:",(y_pred==y_total).mean())

def logistic(z):
    return 1 / (1+np.exp(-z))

def cross_entropy(y_h, y):
    return (- y * np.log(y_h)) - (1-y) * (1  - np.log(y_h))

num_epoch = 1000
learning_rate = 0.01

weight = np.zeros(3)
x_train = np.hstack([x_total,np.ones((x_total.shape[0],1))])

loss_list = []
for i in range(num_epoch):
    y_pred = logistic(np.dot(x_train,weight))
    loss = cross_entropy(y_pred,y_total).mean()
    loss_list.append(loss)

    gradient = (x_train * np.tile((y_pred - y_total).reshape([-1,1]),3)).mean(axis=0)
    weight = weight - learning_rate * gradient

y_pred = np.where(np.dot(x_train,weight)>0,1,0)
print("accuracy:",(y_pred==y_total).mean())

plt.subplot(1,2,1)
plt.plot(np.arange(num_epoch),loss_list)
plt.subplot(1,2,2)
plot_x = np.linspace(-1,1,100)
plot_y = -(weight[0]*plot_x + weight[2])/weight[1]
plt.scatter(x_total[pos_index,0],x_total[pos_index,1],marker='o',c='r')
plt.scatter(x_total[neg_index,0],x_total[neg_index,1],marker='x',c='b')
plt.plot(plot_x,plot_y,c='g')
plt.show()