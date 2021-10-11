'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
import mxnet as mx
import gluoncv

import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay

import numpy as np
import os
from tvm.contrib import utils
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.download import download_testdata
from PIL import Image

cnt_conv_num = 0
network_dict = {"resnet18":"ResNet18_v1b",
                "resnet34":"ResNet34_v1b",
                "resnet50":"ResNet50_v1b",
                "resnet101":"ResNet101_v1b",
                "resnet152":"ResNet152_v1b",
                "vgg11":"VGG11",
                "vgg13":"VGG13",
                "vgg16":"VGG16",
                "vgg19":"VGG19",
                "vgg11_bn":"VGG11_bn",
                "vgg13_bn":"VGG13_bn",
                "vgg16_bn":"VGG16_bn",
                "vgg19_bn":"VGG19_bn",
                "densenet121":"DenseNet121",
                "InceptionV3":"InceptionV3",}

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def benchmark(network, batch_size, warmup=100, batches=400):
    mx.random.seed(0)
    sample = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,3,224,224))
    if network=="InceptionV3":
        sample = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,3,300,300))
    ctx = mx.cpu()

    net = gluoncv.model_zoo.get_model(network_dict[network], pretrained=True)
    net.hybridize(static_alloc=True, static_shape=True)
    net.optimize_for(sample, backend='MKLDNN', static_alloc=True, static_shape=True)
    net(sample)
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
        out = net(sample)
        out.wait_to_read()
    with_fuse_fps = batches * batch_size / (time.time() - tic)
    print('{}: FUSED: {} FPS'.format(network, with_fuse_fps))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "vgg11", "vgg13", "vgg16", "vgg19", 
                "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                "densenet121", "InceptionV3", "all"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")

    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batches", type=int, default=400)
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                    "vgg11", "vgg13", "vgg16", "vgg19", 
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                    "densenet121", 
                    "InceptionV3"]
    else:
        networks = [args.network]

    for network in networks:
        benchmark(network, args.batch_size, \
        warmup=args.warmup, batches=args.batches)
