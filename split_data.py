from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import hashlib
from PIL import Image
from shutil import copyfile
import random

CROPPED_DIR = "./cropped"
TRAIN_DIR = "./training"
TEST_DIR = "./test"
VALIDATION_DIR = "./validation"
VALIDATION_NUM = 10
TEST_NUM = 10
TRAIN_NUM = 68

act = list(set([a.split("\n")[0].strip() for a in open("subset_actors.txt").readlines()]))

def get_cropped_list():
    """
    Get the list of file names of all images in cropped folder
    """
    act_dict = dict()
    if not os.path.isdir(CROPPED_DIR):
        print "Directory \"./cropped\" does not exist."
        return

    files = [ f for f in os.listdir(CROPPED_DIR) ]

    for p in act:
        act_name = p.split()[1].lower()
        act_dict[act_name] = []
        for file in files:
            if file.split('_')[0] == act_name:
                act_dict[act_name].append(file)

    return act_dict

def validate_act_list(act_list):
    for key,val in act_list.iteritems():
        if len(val) < TRAIN_NUM + VALIDATION_NUM + TEST_NUM:
            return False
    return True

def copy_files(flist, target):
    for file in flist:
        copyfile(os.path.join(CROPPED_DIR, file), os.path.join(target, file))

def split_list(act_list):
    if os.path.isdir(TRAIN_DIR) or os.path.isdir(TEST_DIR) or os.path.isdir(VALIDATION_DIR):
        print "Training/Test/Validation directories already exist. Abort splitting"
        return

    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)
    os.makedirs(VALIDATION_DIR)

    for key, val in act_list.iteritems():
        random.shuffle(val)
        copy_files(val[:TEST_NUM], TEST_DIR)
        copy_files(val[TEST_NUM:TEST_NUM+VALIDATION_NUM], VALIDATION_DIR)
        copy_files(val[TEST_NUM+VALIDATION_NUM:TEST_NUM+VALIDATION_NUM+TRAIN_NUM], TRAIN_DIR)

def part1_split(act_list):
    if os.path.isdir(TRAIN_DIR) or os.path.isdir(TEST_DIR) or os.path.isdir(VALIDATION_DIR):
        print "Training/Test/Validation directories already exist. Abort splitting"
        return

    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)
    os.makedirs(VALIDATION_DIR)

    for key, val in act_list.iteritems():
        if key in ["baldwin", "carell"]:
            random.shuffle(val)
            copy_files(val[:TEST_NUM], TEST_DIR)
            copy_files(val[TEST_NUM:TEST_NUM+VALIDATION_NUM], VALIDATION_DIR)
            copy_files(val[TEST_NUM+VALIDATION_NUM:TEST_NUM+VALIDATION_NUM+TRAIN_NUM], TRAIN_DIR)

def split_data(train_num_ow=None):
    global TRAIN_NUM
    if train_num_ow:
        TRAIN_NUM = train_num_ow

    act_list = get_cropped_list()
    if validate_act_list(act_list):
        split_list(act_list)


if __name__ == "__main__":
    split_data()