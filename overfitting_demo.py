import shutil
import os

import matplotlib.pyplot as plt

import split_data
import linear_regression

TRAIN_DIR = "./training"
TEST_DIR = "./test"
VALIDATION_DIR = "./validation"

def main():
    x = []
    valid = []
    test = []
    for i in xrange(2, 68, 5):
        print "Current training set size: ", i

        shutil.rmtree(TRAIN_DIR, ignore_errors=True) 
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        shutil.rmtree(VALIDATION_DIR, ignore_errors=True)

        split_data.split_data(i)

        theta = linear_regression.train()
        x.append(i)
        valid.append(linear_regression.validation(theta))
        test.append(linear_regression.eval(theta))
    plt.plot(x, valid)
    plt.plot(x, test)
    plt.ylabel('Accuracy')
    plt.xlabel('Training Set Size')
    plt.show()
        

if __name__ == "__main__":
    main()