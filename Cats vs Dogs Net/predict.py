import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()

# Size of images
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


mean_blob = caffe_pb2.BlobProto()
with open('C:\\Users\\Mikhail\\PycharmProjects\\Net\\mean.binaryproto', 'rb') as f:
    data = f.read()
    mean_blob.ParseFromString(data)
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

net = caffe.Net('C:\Users\Mikhail\PycharmProjects\Net\model\caffenet_deploy.prototxt',
                'C:\Users\Mikhail\PycharmProjects\Net\model_iter_15000.caffemodel',
               caffe.TEST)
#classifier = caffe.Classifier('C:\Users\Mikhail\PycharmProjects\Net\model\caffenet_deploy.prototxt',
#                              'C:\Users\Mikhail\PycharmProjects\Net\model_iter_15000.caffemodel',
#            image_dims=(128, 128, 3), mean=mean_array)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2, 0, 1))


test_img_paths = [img_path for img_path in glob.glob("C:\Users\Mikhail\PycharmProjects\Net\\test1\*jpg")]

# Making predictions
test_ids = []
preds = []
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    #prediction = classifier.predict([img])
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    print pred_probas.argmax()
    print '-------'

with open("C:\Users\Mikhail\PycharmProjects\Net\submission_model_1.csv", "w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i]) + "," + str(preds[i]) + "\n")
f.close()