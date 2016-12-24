"""
XOR问题一个Nonlinear regression解答.
Chapter6 Page174-177
"""
import numpy as np
import tensorflow as tf

# Data: input and output
X = np.mat([[0,0],[0,1],[1,0],[1,1]]) # formula (6.7) 输入为4*2,训练集样本数为4，特征数为2：x1,x2
# X = np.hstack((np.mat(np.ones((4,1))),X)) # 为X添加一个对应于偏置项b的特征x0，值均为1.
Y = np.mat([[0],[1],[1],[0]])

W = np.mat([[1,1],[1,1]])     # formula (6.4)  shape (2,2)
c = np.mat([0,-1])            # formula (6.5)  shape should be (1,2) not (2,1)
w = np.mat([1,-2]).T          # formula (6.6)  shape (2,1)
XW = X.dot(W)                 # formula (6.8)  shape: (4,2)*(2,2) = (4,2)
b = 0
def ReLU(X):
  # return np.maximum(X, 0)
  # return (abs(X) + X)/2.
  if isinstance(X, np.matrix):    # different element-wise product method for matrix and ndarray in numpy
    return np.multiply(X, (X>0))  # seems to be the fastest
  else: # isinstance(X, np.ndarray) or others
    return 1*(X>0)

#===============================================================================
# def dReLU(X):
#   if isinstance(X, np.matrix):
#     return np.multiply(1, (X>0))
#   else: # isinstance(X, np.ndarray) or others
#     return 1*(X>0)
#===============================================================================

def f(x,W,c,w,b):                 # formula (6.3)  numpy API
  return ReLU(x.dot(W)+c).dot(w) + b

def np_solve():
  return f(X,W,c,w,b)

def tf_f(x,W,c,w,b):              # formula (6.3) tensorflow API
  product1 = tf.matmul(x,W)
  product1 = tf.add(product1, c)
  product2 = tf.nn.relu(product1)
  product2 = tf.matmul(product2, w)
  product = tf.add(product2, b)
  return product
  
def tf_solve():
  tf_X = tf.constant(X, dtype=tf.float64)
  tf_W = tf.constant(W, dtype=tf.float64)
  tf_c = tf.constant(c, dtype=tf.float64)
  tf_w = tf.constant(w, dtype=tf.float64)
  tf_b = tf.constant(b, dtype=tf.float64)
  product = tf_f(tf_X, tf_W, tf_c, tf_w, tf_b)
  # using tf.Session()
  with tf.Session() as sess:
    y_ = sess.run(product)
  # or using tf.InteractiveSession
  # sess = tf.InteractiveSession()
  # y_ = product.eval()
  # sess.close()
  return y_
  
if __name__ == "__main__":
  print("using numpy")
  np_y_ = np_solve()
  print("prediction of Y:\n{}".format(np_y_))
  print("\n")
  print("using tensorflow")
  tf_y_ = tf_solve()
  print("prediction of Y:\n{}".format(tf_y_))
  
  