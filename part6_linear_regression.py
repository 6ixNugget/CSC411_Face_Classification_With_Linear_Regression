from __future__ import division

import numpy as np
import os
from numpy import *
from numpy.linalg import norm
import fnmatch
from scipy.misc import imread
from numpy.linalg import inv
from numpy import linalg as LA
from scipy import misc
from matplotlib.pyplot import *


TRAIN_DIR = "./training"
TEST_DIR = "./test"
VALIDATION_DIR = "./validation"
SAVE_PATH = "./theta_visual"
WEIGHT_SAVE = "./trained_weights.npy"

IMAGE_SHAPE = 32
act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
male_act = ["baldwin", "carell", "hader"]
female_act = ["harmon", "gilpin", "bracco"]

def build_input(DIR):
    if not os.path.isdir(DIR):
        print "Directory does not exist."
        return
    
    x, y = [], []

    files = [ f for f in os.listdir(DIR) ]
    for file in files:
        tag = file.split('_')[0]
        if tag not in act:
            print "Corrupted data, abort training"
            return

        x.append(misc.imread(os.path.join(DIR, file)).flatten()/255.)

        onehot = np.zeros(len(act))
        onehot[act.index(tag)] = 1
        y.append(onehot)

    x = np.array(x)
    y = np.array(y)

    return x.T, y.T

def f(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return sum((y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return -2*dot(x,(y-dot(theta.T, x)).T)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10
    prev_t = init_t - 10 * EPS
    t = init_t.copy()

    max_iter = 5000 * 18
    ite  = 0
    while norm(t - prev_t) >  EPS and ite < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if ite % 5000 == 0:
            validation_accuracy = validation(t)
            print "Iter", ite
            print "Gradient: ", df(x, y, t), "\n"
            print "Loss: ", f(x, y, t), "\n"
            print "Validation Accuracy ", validation_accuracy, "\n"
        ite += 1
    return t

def train():
    x, y = build_input(TRAIN_DIR)
    theta0 = np.zeros((IMAGE_SHAPE*IMAGE_SHAPE+1, len(act)))
    return grad_descent(f, df, x, y, theta0, 5e-6)

def validation(theta):
    x, y = build_input(VALIDATION_DIR)
    x = vstack((ones((1, x.shape[1])), x))
    ans_tag = np.argmax(dot(theta.T, x), axis=0)
    target_tag = np.argmax(y, axis=0)
    accuracy = sum(ans_tag==target_tag)/len(target_tag)
    return accuracy

def eval(theta):
    x, y = build_input(TEST_DIR)
    x = vstack((ones((1, x.shape[1])), x))
    ans_tag = np.argmax(dot(theta.T, x), axis=0)
    target_tag = np.argmax(y, axis=0)
    accuracy = sum(ans_tag==target_tag)/len(target_tag)
    return accuracy

def finite_diff():
    x, y = build_input(TEST_DIR)
    theta = np.load(WEIGHT_SAVE)

    h = 0.000001
    i = 555
    j = 4

    theta_add_h = theta.copy()
    theta_minus_h = theta.copy()

    theta_add_h[i][j] = theta_add_h[i][j]+h
    theta_minus_h[i][j] = theta_minus_h[i][j]-h

    print (f(x, y, theta_add_h) - f(x, y, theta_minus_h))/(2*h)
    print df(x, y, theta)[i][j]


if __name__ == "__main__":
    # theta = train()

    # print "The final accuracy is ", eval(theta)

    # for i in xrange(theta.shape[1]):
    #     imsave(SAVE_PATH+'_'+str(i)+'.jpg', reshape(theta.T[i][1:], (IMAGE_SHAPE,IMAGE_SHAPE)), cmap=cm.RdBu)

    # np.save(WEIGHT_SAVE, theta)
    finite_diff()
