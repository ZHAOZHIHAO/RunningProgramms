#!/usr/bin/env python
#give the path of caffe
import sys
import os
import os.path as osp
sys.path.insert(0,'..')
caffe_root="/caffe"
import time

import numpy as np
from PIL import Image
import scipy.misc
sys.path.insert(0, os.path.join(caffe_root,'/python'))
#sys.path.append("./layers") # the datalayers we will use are in this directory.
import caffe

import cv2
from os import listdir
from os.path import isfile, join

caffe.set_device(0)
caffe.set_mode_gpu()
# http://cs231n.stanford.edu/reports/2016/pdfs/255_Report.pdf
labels = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61", "Backpack", "CarryingOther", "Casual lower", "Casual upper", "Formal lower", "Formal upper", "Hat", "Jacket", "Jeans", "Leather Shoes", "Logo", "Long hair", "Male", "Messenger Bag", "Muffler", "No accessory", "No carrying", "Plaid", "PlasticBags", "Sandals", "Shoes", "Shorts", "Short Sleeve", "Skirt", "Sneaker", "Stripes", "Sunglasses", "Trousers", "Tshirt", "UpperOther", "V-Neck"] 
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 10)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1

def safe_divide(x,y):
    if y == 0.0: return 0.0
    else:        return x/y


def load_mean(fname_bp):
    """
    Load mean.binaryproto file.
    """
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(fname_bp , 'rb' ).read()
    blob.ParseFromString(data)
    return np.array(caffe.io.blobproto_to_array(blob))


def run_evaluation(fname_model, fname_weights, fname_testdata, path_out, fname_mean, path_dataset, layer_pred="prob-attr", batchsize=2):
    tstart = time.time()

    # read testdata
    with open(fname_testdata, 'r') as f:
        testdata = [(l.strip().split()[0], l.strip().split()[1].split(',')) for l in f.readlines()]

    # generate attribute prediction file
    if not osp.exists(path_out): os.makedirs(path_out)
    fname_features = osp.join(path_out, "predictions.npy")
    #fname_angles = osp.join(path_out, "angles.npy") 
    if not osp.exists(fname_features):
        # load mean
        mean = load_mean(fname_mean)[0,:]
        mean = mean[:,:,::-1]           # wrong, but consistant for training and testing (channel is first dimension)
        mean = mean.transpose((1,2,0))

        # create net
        net = caffe.Net(fname_model, fname_weights, caffe.TEST)
        bs = batchsize
        net.blobs['data'].reshape(bs, 3, 227, 227)
        net.reshape()

        # predict
        features = []
        #angles = []
        image_folder = './images'
        images_path = [join(image_folder, f) for f in listdir(image_folder) if isfile(join(image_folder, f))]
        for i in xrange(0,len(images_path),bs):
            batch = []
            image_names = []
            for j in range(bs):
                idx = min(i+j, len(images_path)-1)
                fname_image = images_path[idx]
                img = np.asarray(Image.open(fname_image))
                img = scipy.misc.imresize(img, (256,256), interp='bicubic')
                img = img[:, :, :3]
                img = np.subtract(img, mean)
                #img = scipy.misc.imresize(img, (227,227), interp='bicubic')
                img = img[15:15+227, 15:15+227, :]
                img = img[:,:,::-1]
                img = img.transpose((2,0,1))
                batch.append(img)
                image_names.append(osp.join(path_dataset, fname_image))

            batch = np.asarray(batch)
            net.blobs['data'].data[...] = batch
            o = net.forward()

            for j in range(bs):
                if i+j >= len(images_path): break
                features.append(np.copy(net.blobs[layer_pred].data.squeeze()[j,:]))
                #angles.append(np.copy(net.blobs['prob-angle'].data.squeeze()[j,:]))

                # add by zz
                #np.set_printoptions(precision=3)
                np.set_printoptions(suppress=True, precision=3)
                probabilities = np.copy(net.blobs[layer_pred].data.squeeze()[j,:])
                idx = min(i+j, len(images_path)-1)
                show_img_name = images_path[idx]
                print show_img_name
                img = cv2.imread(show_img_name)
                h, w, c = img.shape
                new_h = max(800, h)
                img1 = np.zeros((new_h, w+200, c), np.uint8)
                img1[:h, 200:200+w] = img
                attr_idx = 0
                for label_name, label_probability in zip(labels, probabilities):
                    attr_idx += 1
                    text = label_name + ": %.2f" % (label_probability)
                    #print label_name, "%.2f" % (label_probability)
                    bottomLeftCornerOfText = (10, (attr_idx + 1) * 20)
                    cv2.putText(img1, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType) 
                cv2.imwrite('out' + str(int(idx)) + '.jpg', img1)
                

sw = 1

if sw == 1:
    fname_model    = "../PETA/deploy_peta.prototxt"
    fname_weights  = "../vespa-peta_iter_12000.caffemodel"
    fname_testdata = "../generated/PETA_test_list.txt"
    path_dataset   = "../PETA_dataset/"
    fname_mean     = "../generated/peta_mean.binaryproto"
    path_out       = "../eval_peta/"
    run_evaluation(fname_model, fname_weights, fname_testdata, path_out, fname_mean, path_dataset)

elif sw == 2:
    fname_model    = "../RAP/deploy_rap.prototxt"
    fname_weights  = "../snapshots/vespa-rap_iter_16000.caffemodel"
    fname_testdata = "../generated/RAP_test_list.txt"
    path_dataset   = "/cvhci/users/aschuman/datasets/RAP/"
    fname_mean     = "../generated/rap_mean.binaryproto"
    path_out       = "../eval_rap/"
    run_evaluation(fname_model, fname_weights, fname_testdata, path_out, fname_mean, path_dataset)
