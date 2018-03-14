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

UNCROPPED_DIR = "./uncropped"
CROPPED_DIR = "./cropped"

act = list(set([a.split("\n")[0].strip() for a in open("subset_actors.txt").readlines()]))

def crop_and_grey():
    '''
    Crop images and convert to greyscale
    '''
    if os.path.isdir(CROPPED_DIR):
        print "Directory \"./cropped\" already exists, abort cropping."
        return

    os.makedirs(CROPPED_DIR)
    files = [ f for f in os.listdir(UNCROPPED_DIR) if os.path.isfile(os.path.join(UNCROPPED_DIR, f)) ]
    for f in files:
        try:
            print f
            bounding_box = f.split('_')[2].split(',')
            box = (int(bounding_box[0]), int(bounding_box[1]),
                    int(bounding_box[2]), int(bounding_box[3]))
            print box
            im = Image.open(os.path.join(UNCROPPED_DIR, f)).convert('L')
            cropped = np.asarray(im.crop(box))
            while cropped.shape[0] > 64:
                cropped = imresize(cropped, .5)
            cropped = imresize(cropped, [32,32])
            img = Image.fromarray(cropped)
            img.save(os.path.join(CROPPED_DIR, f.split()[0]))
        except:
            continue

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            

def download_images():
    """
    Download images if not present.
    """
    if not os.path.isdir(UNCROPPED_DIR):
        os.makedirs(UNCROPPED_DIR)
        for a in act:
            name = a.split()[1].lower()
            i = 0
            for line in open("faces_subset.txt"):
                if a in line:
                    filename = name+'_'+str(i)+'_'+line.split('\t')[4]+'_'+'.'+line.split('\t')[3].split('.')[-1]
                    #A version without timeout (uncomment in case you need to 
                    #unsupress exceptions, which timeout() does)
                    #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                    #timeout is used to stop downloading images which take too long to download
                    timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                    if not os.path.isfile("uncropped/"+filename):
                        continue

                    print filename
                    i += 1
    else:
        print "Directory \"./uncropped\" already exists, abort downloading."

if __name__ == "__main__":
    download_images()
    crop_and_grey()
