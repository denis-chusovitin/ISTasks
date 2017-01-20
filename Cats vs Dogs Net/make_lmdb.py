import numpy as np
import cv2
from caffe.proto import caffe_pb2
import lmdb
from os import listdir
import random

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

train_lmdb="C:\Users\Mikhail\PycharmProjects\Net\\train_lmdb"
test_lmdb="C:\Users\Mikhail\PycharmProjects\Net\\test_lmdb"
test1_lmdb = "C:\Users\Mikhail\PycharmProjects\Net\\test1_lmdb"
train_data_path="train"
train_data=listdir(train_data_path)
random.shuffle(train_data)
test_data_path="test1"
test_data=listdir(test_data_path)
random.shuffle(test_data)

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def make_datum(img, label):

    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def make_train_lmdb():
    in_db = lmdb.open(train_lmdb, map_size=17179869184)
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(train_data):
            if in_idx % 6 == 0:
                continue
            img = cv2.imread(train_data_path + "\\" + img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

def make_test_lmdb():
    in_db = lmdb.open(test_lmdb, map_size=2147483648)
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(train_data):
            if in_idx % 6 != 0:
                continue
            img = cv2.imread(train_data_path + "\\" + img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

make_train_lmdb()
make_test_lmdb()