import time
import mxnet as mx
import warnings
warnings.filterwarnings("ignore")
from mxnet.gluon.model_zoo.vision import *
import gluoncv
model_dict = {'resnet50_v1': resnet50_v1}#'mobilenet_v2_1_0': mobilenet_v2_1_0, 
network_dict = {"resnet50":"ResNet50_v1b"}
def benchmark(batch_size=1, batches=400, warmup=100):
    mx.random.seed(0)
    sample = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,3,224,224))
    ctx = mx.cpu()

    for model_name in model_dict.keys():
        # net = model_dict[model_name](pretrained=True)
        #net.hybridize(static_alloc=True, static_shape=True)
        # net(sample)
        # for i in range(batches+warmup):
        #     if i == warmup:
        #         tic = time.time()
        #     out = net(sample)
        #     out.wait_to_read()
        # without_fuse_fps = batches * batch_size / (time.time() - tic)

        #net = model_dict[model_name](pretrained=True)
        #net = mx.gluon.model_zoo.vision.get_resnet(1, 50, pretrained=True)
        net = gluoncv.model_zoo.get_model(network_dict['resnet50'], pretrained=True)
        net.hybridize(static_alloc=True, static_shape=True)
        # print(#net)
        net.optimize_for(sample, backend='MKLDNN', static_alloc=True, static_shape=True)
        net(sample)
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = net(sample)
            out.wait_to_read()
        with_fuse_fps = batches * batch_size / (time.time() - tic)
        # net = model_dict[model_name](pretrained=True)
        # calib_data_loader = mx.gluon.data.DataLoader(sample, 128)
        # qnet = mx.contrib.quantization.quantize_net(net, calib_mode='naive', calib_data=calib_data_loader)
        # qnet.hybridize(static_alloc=True, static_shape=True)
        # qnet(sample)
        # for i in range(batches+warmup):
        #     if i == warmup:
        #         tic = time.time()
        #     out = qnet(sample)
        #     out.wait_to_read()
        # quantized_fps = batches * batch_size / (time.time() - tic)

        print('{}: FUSED: {} FPS'.format(model_name, with_fuse_fps))

benchmark(1)
