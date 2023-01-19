# coding=utf-8
import numpy as np


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta

    # theta:k,n     x: m,n      y:k,m

    m, n = x.shape

    for i in range(iters):

        # 1.计算预测值
        y_hat = softmax(np.matmul(theta, x.T))

        # 2.计算损失值
        loss = -1/m * np.sum(y*np.log(y_hat))
        print(f'Iter{i+1}Loss:', loss)

        # 3.梯度下降、更新参数
        theta -= alpha/m * np.matmul((y_hat - y), x)

    return theta


def softmax(x):
    # X:(10,m)
    return np.exp(x) / np.sum(np.exp(x), axis=0)    # 对每列求和 axis=0





    
