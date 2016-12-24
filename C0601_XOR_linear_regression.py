""" 
尝试XOR问题的线性模型求解
分别使用numpy和tensorflow库通过求解norm equation 的方式来计算XOR问题线性回归模型的参数
Author:qqiangye@gmail.com
Date:2016-12-24
"""
# 深度前向传播神经网络(Deep Feedforward Neural Network)，也经常被称为前向传播神经网络(feedforward neural network)
# 或多层感知器(multilayer perceptrons, MLPs)，是一种极其典型的深度学习模型。
# XOR problem:
import numpy as np
import tensorflow as tf
# Data: input and output
X = np.mat([[0,0],[0,1],[1,0],[1,1]]) # 输入为4*2,训练集样本数为4，特征数为2：x1,x2
X = np.hstack((np.mat(np.ones((4,1))),X)) # 为X添加一个对应于偏置项b的特征x0，值均为1.
Y = np.mat([[0],[1],[1],[0]])

def np_solve():
  # norm equation 求解参数，对于这里的X和Y,参数是一个3*1的向量，
  # 分别对应于偏置项b,及特征x0，x1的权重w1,w2.
  # norm equation: X.dot(theta) = Y
  # then the solution for theta is as follow:
  return X.T.dot(X).I.dot(X.T).dot(Y)  


def tf_solve():
  tfX = tf.constant(X, dtype=tf.float64)
  tfY = tf.constant(Y, dtype=tf.float64)
  tfXT = tf.matrix_transpose(tfX)
  product1 = tf.matmul(tfXT,tfX)
  product1_inverse = tf.matrix_inverse(product1)
  product2 = tf.matmul(product1_inverse,tfXT)
  product3 = tf.matmul(product2,tfY)
  # using tf.InteractiveSession()
  sess = tf.InteractiveSession()
  valuex, valuey, theta = tfX.eval(), tfY.eval(), product3.eval()
  sess.close()
  # or using tf.Session()
  # with tf.Session() as sess:
  #   valuex,valuey,theta = sess.run((tfX,tfY,product3))
  return valuex,valuey,theta

if __name__ == "__main__":
  print("using numpy")
  theta = np_solve()
  print("X:\n{}\n\nY:\n{}\n\ntheta:\n{}".format(X, Y, theta))
  print("\n\n")
  print("using tensorflow")
  valuex, valuey, theta = tf_solve()
  print("X:\n{}\n\nY:\n{}\n\ntheta:\n{}".format(valuex,valuey,theta))