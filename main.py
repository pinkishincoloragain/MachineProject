import sys
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data     #텐서플로우에서 제공하는 데이터 가져오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 데이터 저장

x = tf.placeholder(tf.float32, [None, 784]) # 여기서 None은 해당 차원의 길이가 어떤 길이든 가능하다는 것
W = tf.Variable(tf.zeros([784, 10]))        # 0으로 초기화, 784차원의 이미지 픽셀을 숫자마다 weight를 따로 생성
b = tf.Variable(tf.zeros([10]))             # W에 더하는 Bias이기 때문에 위해 10차원으로 생성
y = tf.nn.softmax(tf.matmul(x, W) + b)      # softmax 적용 (예측값)
y_= tf.placeholder(tf.float32, [None, 10]) # 실제값 (즉 1,2,3 등과 같은 실제값)

