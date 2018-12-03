#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jonathon Headley
"""
from keras.datasets import mnist

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from support import *
import time
import sys
percTrain = float(sys.argv[1])#.01
percTest = float(sys.argv[2])#.05


def computeK(x, y):
    out = np.zeros((x.shape[0], y.shape[0]), dtype='float64')
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            out[i][j] = np.exp(np.sum(-1.0*np.square(np.subtract(x[i], y[j]))))
    return out.astype('float64')


def forget_a_few(y, percToForget):
    numToKeep = int(y.shape[0] * (1-percToForget))
    y[numToKeep:] = 0
    return y


def accuracyNP(y_true, y_pred):
    sCnt = y_true.shape[0]
    errors = -np.sign(-1+np.sign(np.multiply(y_true, y_pred)))
    # print(errors)
    # print([1 if x[0]==1 else -1 for x in errors])
    labeled=[1 if x[0] == 1 else -1 for x in errors]
    return labeled,accuracy_score(labeled, y_true)


def separateClasses(x_train, y_train, classValue1, classValue2):
    x_class1 = []
    x_class2 = []
    y_class1 = []
    y_class2 = []
    for i in range(0, len(x_train)):
        if y_train[i] in classValue1:
            x_class1.append(x_train[i])
            y_class1.append(1)
        elif y_train[i] in classValue2:
            x_class2.append(x_train[i])
            y_class2.append(-1)
        else:
            print("why was", x_train[i], "with class",
                  y_train[i], "not placed...")
    return x_class1, x_class2, y_class1, y_class2


def norm(y, class1, class2):
    y_out = []
    for i in range(0, len(y)):
        if y[i] in classValue1:
            y_out.append(1)
        elif y[i] in classValue2:
            y_out.append(-1)
    return np.array(y_out).astype('float64').reshape(y.shape[0], 1)


np.random.seed(123)

start_time = time.time()

classValue1 = [0, 1, 2, 3, 4]
classValue2 = [5, 6, 7, 8, 9]
(x_train_o, y_train_o), (x_test_o, y_test_o) = mnist.load_data()


numToTrainOn = int(percTrain*x_train_o.__len__())
numToTestOn = int(percTest*x_test_o.__len__())
print("training with", numToTrainOn,
      "images and testing with", numToTestOn, "images")
x_train = np.array(x_train_o[:numToTrainOn])
y_train = np.array(y_train_o[:numToTrainOn])
x_test = np.array(x_test_o[:numToTestOn])
y_test = np.array(y_test_o[:numToTestOn])
x_train = flat_norm(x_train)
x_test = flat_norm(x_test)
y_train = norm(y_train, classValue1, classValue2)
y_test = norm(y_test, classValue1, classValue2)
print("done normalizing")

np.set_printoptions(linewidth=250)
percToForget = .9


y_train = forget_a_few(y_train, percToForget)


fCnt = len(x_train[0])
n_train = len(x_train)
# START OF LEARNING

# number of epchos. 1 epoch is when all training data is seen
n_epochs = 100

# define variables for tensorflow

# define and initialize shared variables
# (the variable persist, they encode the state of the classifier throughout learning via gradient descent)
# w is the feature weights, a [fCnt x 1] vector
initialC = np.random.rand(n_train, 1)*y_train  # *0.0*y_train
c = tf.Variable(initialC, name="w", dtype='float64')

# b is the bias, so just a single number
initialB = 0.0
b = tf.Variable(initialB, name="b", dtype='float64')


# npK=np.dot(x_train,np.transpose(x_train))
# npKtrainVStest=np.dot(x_test,np.transpose(x_train))
# npK=np.exp(-1.0*np.reduce_sum(np.square(x_train - x_train)))
npK = computeK(x_train, x_train)
npKtrainVStest = computeK(x_test, x_train)

K = tf.constant(npK, dtype=tf.float64, name='K')
y = tf.constant(y_train, dtype=tf.float64, name='y')


def computeG(x, y, t=1):
    out = np.zeros((x.shape[0], y.shape[0]), dtype='float64')
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            out[i][j] = np.exp(
                (np.sum(-1.0*np.square(np.subtract(x[i], y[j]))))/t)
    return out.astype('float64')


def computeD(g):
    d = np.zeros(g.shape)
    shape = g.shape
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if i == j:
                d[i][j] = np.sum(g[i])
    return d


g = computeG(x_train, x_train)
print("g is", g)
a = g
d = computeD(g)
L = d-a
print("L is", L)
# set up new variables that are functions/transformations of the above
# predicted class for each sample (a vector)
# tf.matmul(x,w) is a vector with #samples entries
# even though b is just a number, + will work (through "broadcasting")
# b will be "replicated" #samples times to make both sides of + have same dimension
# thre result is a vector with #samples entries
# h = tf.matmul(K, a)+b

print("calculating h")
transposedC = tf.transpose(c)
secondTerm = .5*transposedC @ K @ c + transposedC @ K @ L @ K @ c

# performs a single loop over the kernel?


def h_i(i, c, y, k):
    h = tf.reduce_sum(tf.multiply(tf.multiply(c, y),
                                  tf.reshape(k[i], (-1, 1))))+b
    return tf.log(1.0+tf.exp(-y*h))

    # return tf.log(1.0+tf.exp(-tf.multiply(y, h)))
# doubleSum=tf.Variable(0,name="doubleSum",dtype=tf.float64)


iTF = tf.constant(0, name="iTF", dtype="int32")
summTF = tf.constant(np.zeros((n_train, 1)), name="summTF", dtype="float64")
endIterationsTF = tf.constant(
    int(n_train-(n_train*percToForget)), name="endIterationsTF", dtype="int32")


def h_cond(i, summ, endIterations):
    return tf.less(i, endIterations)


def h_body(i, summ, endIterations):
    return [tf.add(i, 1), tf.add(summ, tf.add(h_i(i, c, y, K), secondTerm)), endIterations]


_, h, _ = tf.while_loop(
    h_cond,
    h_body,
    [iTF, summTF, endIterationsTF]
)
# loss (square error of prediction) for each sample (a vector)
loss = h


# risk over all samples (a number)
risk = tf.reduce_mean(loss)

# define which optimizer to use

# optimizer = tf.train.AdamOptimizer(0.01)
optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(risk)

# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def predict(k, a, b):
    mult = k @ a + b
    return mult


c_value = c.eval(session=sess)
b_value = b.eval(session=sess)
predTrain = predict(npK, c_value, b_value)

_,accTr = accuracyNP(y_train, predTrain)

predTest = predict(npKtrainVStest, c_value, b_value)
# print(y_test)
# print(predTest)
_,accTe = accuracyNP(y_test, predTest)
print('Stats: %f %f %f %f %f' % (100*accTr, 100*accTe,
                                 np.mean(np.abs(c_value)), np.mean(c_value), b_value))

# start the iterations of training
# 1 epoch == all data samples were presented
for i in range(0, n_epochs):
    train.run(session=sess)
    c_value = c.eval(session=sess)
    b_value = b.eval(session=sess)
    # print("A value is",np.transpose(c_value))

    secondTerm_value = secondTerm.eval(session=sess)
    k = K.eval(session=sess)
    # print(k)
    # print(np.transpose(c_value))
    # print(y_value)
    print("risk:", risk.eval(session=sess))

    # no L1 regularization in this version
    # zero-values for a feature weight may not be in the subspace spanned by training samples, i.e. impossible to get with tf.matmul(K,c)

    predTrain = predict(npK, c_value, b_value)
    _,accTr = accuracyNP(y_train, predTrain)

    predTest = predict(npKtrainVStest, c_value, b_value)
    _,accTe = accuracyNP(y_test, predTest)
    print('Stats: %f %f %f %f %f' %
          (100*accTr, 100*accTe, np.mean(np.abs(c_value)), np.mean(c_value), b_value))

est_a = c.eval(session=sess)
est_b = b.eval(session=sess)
# print("A is",est_a,"b is",est_b)
predTest = np.dot(npKtrainVStest, est_a)+est_b
# print("predTest",predTest)
labeled,acc = accuracyNP(y_test, predTest)
print('Accuracy: %f' % (100*acc))


# Analysis
numClass0 = 0
numClass1 = 0
y_test_class0 = []
y_test_class1 = []
test_class = []
diff=time.time()-start_time

print("That took", "{:.2f}".format(
    round(diff, 2)), "seconds to run")


print("Accuracy is", "{0:.2f}".format(round(acc, 4)*100)+"% using",
      y_train.__len__(), "training samples and", y_test.__len__(), "testing samples, each with", fCnt, "features.")
metrics = precision_recall_fscore_support(y_test, labeled, labels=[-1, 1])

print("\t|  Precision\t|  Recall\t|  FScore")
print("--------+---------------+---------------+-----------")
print("Class 1\t| ", "{0:.4f}".format(round(metrics[0][1], 4)), "\t| ", "{0:.4f}".format(
    round(metrics[1][1], 4)), "\t| ", "{0:.4f}".format(round(metrics[2][1], 4)))
print("Class 2\t| ", "{0:.4f}".format(round(metrics[0][0], 4)), "\t| ", "{0:.4f}".format(
    round(metrics[1][0], 4)), "\t| ", "{0:.4f}".format(round(metrics[2][0], 4)))
print()
# first 2 are time and accuracy followed by class 2 precision recall fscore then class 1 precision recall fscore
print(diff,round(acc,4),metrics[0][0],metrics[1][0],metrics[2][0],metrics[0][1],metrics[1][1],metrics[2][1])