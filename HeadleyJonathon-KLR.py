#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jonathon Headley
"""
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from support import *
import time
from tensorboard import summary as summary_lib

percTrain = .01
percTest = .05


def computeK(x, y):
    print("x shape",x.shape)
    print("y shape",y.shape)
    out = np.zeros((x.shape[0],y.shape[0]), dtype='float64')
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            out[i][j] = np.exp(np.sum(-1.0*np.square(np.subtract(x[i],y[j]))))
    return out.astype('float64')


def accuracyNP(y_true, y_pred):
    sCnt = y_true.shape[0]
    errors = -np.sign(-1+np.sign(np.multiply(y_true, y_pred)))
    accuracy = (sCnt-np.sum(errors))/sCnt
    return accuracy


def accuracy01(y_true, y_pred):
    return accuracyNP(y_true-0.5, y_pred-0.5)


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

# x_train = np.array([[1, 2], [1, 2], [1, 2]])
# x_test = np.array([[4, 5], [4, 5]])
# y_train = np.array([1, 0, 1])
# y_test = np.array([0, 1])
y_train = norm(y_train, classValue1, classValue2)
y_test = norm(y_test, classValue1, classValue2)
np.set_printoptions(linewidth=250)
# x_class1, x_class2, y_class1, y_class2 = separateClasses(
#     x_train, y_train, classValue1, classValue2)
# print("Separated train classes")
# x_test_class1, x_test_class2, y_test_class1, y_test_class2 = separateClasses(
#     x_test, y_test, classValue1, classValue2)
# print("Separated test classes")


fCnt = len(x_train[0])
n_train = len(x_train)
n_epochs = 10

# START OF LEARNING

# number of epchos. 1 epoch is when all training data is seen
n_epochs=1500

# define variables for tensorflow

# define and initialize shared variables
# (the variable persist, they encode the state of the classifier throughout learning via gradient descent)
# w is the feature weights, a [fCnt x 1] vector
initialC = np.random.rand(n_train, 1)  # *0.0*y_train
c = tf.Variable(initialC, name="w")

# b is the bias, so just a single number
initialB = 0.0
b = tf.Variable(initialB, name="b", dtype='float64')


# npK=np.dot(x_train,np.transpose(x_train))
# npKtrainVStest=np.dot(x_test,np.transpose(x_train))
# npK=np.exp(-1.0*np.reduce_sum(np.square(x_train - x_train)))
npK = computeK(x_train, x_train)
print("x_test shape", x_test.shape)
print("x_test", x_test)
print("x_train shape", x_train.shape)
print("x_train", x_train)
npKtrainVStest = computeK(x_test, x_train)
print("npk shape", npK.shape)
print("npk", npK)
print("npktrainvstest shape", npKtrainVStest.shape)
print("npKTrainVTest", npKtrainVStest)
K = tf.constant(npK, dtype=tf.float64, name='K')
y = tf.constant(y_train, dtype=tf.float64, name='y')

# set up new variables that are functions/transformations of the above
# predicted class for each sample (a vector)
# tf.matmul(x,w) is a vector with #samples entries
# even though b is just a number, + will work (through "broadcasting")
# b will be "replicated" #samples times to make both sides of + have same dimension
# thre result is a vector with #samples entries
predictions = tf.matmul(K, c)+b
# loss (square error of prediction) for each sample (a vector)
loss = tf.log(1.0+tf.exp(-tf.multiply(y, predictions)))
# risk over all samples (a number)
risk = tf.reduce_mean(loss)

# define which optimizer to use
#optimizer = tf.train.AdamOptimizer(0.001)
optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(risk)

# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

c_value = c.eval(session=sess)
print("c shape", c_value.shape)
b_value = b.eval(session=sess)
predTrain = np.dot(npK, c_value)+b_value

accTr = accuracyNP(y_train, predTrain)

predTest = np.dot(npKtrainVStest, c_value)+b_value
print(y_test)
print(predTest)
print("y_test shape", y_test.shape)
print("predTest shape", predTest.shape)
accTe = accuracyNP(y_test, predTest)
print('Stats: %f %f %f %f %f' % (100*accTr, 100*accTe,
                                 np.mean(np.abs(c_value)), np.mean(c_value), b_value))

# start the iterations of training
# 1 epoch == all data samples were presented
for i in range(0, n_epochs):
    train.run(session=sess)
    c_value = c.eval(session=sess)
    b_value = b.eval(session=sess)
    print("risk:", risk.eval(session=sess))

    # no L1 regularization in this version
    # zero-values for a feature weight may not be in the subspace spanned by training samples, i.e. impossible to get with tf.matmul(K,c)

    predTrain = np.dot(npK, c_value)+b_value
    accTr = accuracyNP(y_train, predTrain)

    predTest = np.dot(npKtrainVStest, c_value)+b_value
    accTe = accuracyNP(y_test, predTest)
    print('Stats: %f %f %f %f %f' % (100*accTr, 100*accTe,
                                     np.mean(np.abs(c_value)), np.mean(c_value), b_value))

est_c = c.eval(session=sess)
est_b = b.eval(session=sess)
predTest = np.dot(npKtrainVStest, est_c)+est_b
print(predTest)
acc = accuracyNP(y_test, predTest)
print('Accuracy: %f' % (100*acc))


# Analysis
numClass0 = 0
numClass1 = 0
y_test_class0 = []
y_test_class1 = []
test_class = []

print("That took", "{:.2f}".format(
    round(time.time()-start_time, 2)), "seconds to run")
print("We predicted we have", numClass0, "images of", classValue1, "'s.  We actually have",
      y_test_class0.__len__(), "images of", classValue1, "'s")
print("We predicted we have", numClass1, "images of", classValue2, "'s.  We actually have",
      y_test_class1.__len__(), "images of", classValue2, "'s")


print("Accuracy is", "{0:.2f}".format(round(acc, 4)*100)+"% using",
      y_train.__len__(), "training samples and", y_test.__len__(), "testing samples, each with", fCnt, "features.")
# metrics = precision_recall_fscore_support(y_test, predTest, labels=[-1, 1])

# print("\t|  Precision\t|  Recall\t|  FScore")
# print("--------+---------------+---------------+-----------")
# print("Class 1\t| ", "{0:.4f}".format(round(metrics[0][1], 4)), "\t| ", "{0:.4f}".format(
#     round(metrics[1][1], 4)), "\t| ", "{0:.4f}".format(round(metrics[2][1], 4)))
# print("Class 2\t| ", "{0:.4f}".format(round(metrics[0][0], 4)), "\t| ", "{0:.4f}".format(
#     round(metrics[1][0], 4)), "\t| ", "{0:.4f}".format(round(metrics[2][0], 4)))
# print()
