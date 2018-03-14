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
NON_ACT_DIR = "./cropped2"
VALIDATION_DIR = "./validation"
SAVE_PATH = "./trained_weights.jpg"
WEIGHT_SAVE = "./trained_weights.npy"

IMAGE_SHAPE = 32
act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
male_act = ["baldwin", "carell", "hader"]
female_act = ["harmon", "gilpin", "bracco"]

non_act = ['radcliffe', 'butler', 'vartan', 'chenoweth', 'drescher', 'ferrera']
male_non_act = ['radcliffe', 'butler', 'vartan']
female_non_act =  ['chenoweth', 'drescher', 'ferrera']

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
        y.append(int(tag in male_act))

    x = np.array(x)
    y = np.array(y)

    return x.transpose(), y

def build_input_non_act(DIR):
    if not os.path.isdir(DIR):
        print "Directory does not exist."
        return
    
    x, y = [], []

    files = [ f for f in os.listdir(DIR) ]
    for file in files:
        tag = file.split('_')[0]
        if tag not in non_act:
            print "Corrupted data, abort training"
            return

        x.append(misc.imread(os.path.join(DIR, file)).flatten()/255.)
        y.append(int(tag in male_non_act))

    x = np.array(x)
    y = np.array(y)

    return x.transpose(), y


def f(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return sum((y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10
    prev_t = init_t - 10 * EPS
    t = init_t.copy()

    max_iter = 5000 * 100
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
    print "training_size", len(y)
    theta0 = np.zeros(IMAGE_SHAPE*IMAGE_SHAPE+1)
    return grad_descent(f, df, x, y, theta0, 1e-6)

def validation(theta):
    x, y = build_input(VALIDATION_DIR)
    x = vstack((ones((1, x.shape[1])), x))
    ans_tag = dot(theta.T, x) > 0.5
    accuracy = sum(ans_tag==y)/len(y)
    return accuracy

def eval(theta):
    x, y = build_input(TEST_DIR)
    x = vstack((ones((1, x.shape[1])), x))
    ans_tag = dot(theta.T, x) > 0.5
    accuracy = sum(ans_tag==y)/len(y)
    return accuracy

def eval_non_act(theta):
    x, y = build_input(NON_ACT_DIR)
    x = vstack((ones((1, x.shape[1])), x))
    ans_tag = dot(theta.T, x) > 0.5
    accuracy = sum(ans_tag==y)/len(y)
    return accuracy

if __name__ == "__main__":
    theta = train()
    
    print "The final accuracy is ", eval(theta)
    print "The final accuracy on non_act", eval_non_act(theta)
    imsave(SAVE_PATH, reshape(theta[1:], (IMAGE_SHAPE,IMAGE_SHAPE)), cmap=cm.RdBu)
    np.save(WEIGHT_SAVE, theta)
