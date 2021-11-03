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
                "InceptionV3":"InceptionV3",
                "MobileNet1.0":"MobileNet1.0",
                "i3d_resnet50_v1_kinetics400":"i3d_resnet50_v1_kinetics400"}

data_dic = {"a":"N",
            "b":"C",
            "c":"H",
            "d":"W",}

weight_dic = {"a":"O",
              "b":"I",
              "c":"H",
              "d":"W",
              "e":"G"}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    def get_shape(tensor):
        if 'Var' in str(type(tensor)):
            return tensor.type_annotation.concrete_shape
        elif 'Constant' in str(type(tensor)):
            return tensor.data.shape
        elif 'TensorType' in str(type(tensor)):
            return tensor.concrete_shape
        else:
            if "pad" in tensor.op.name:
                return tensor.type_args[0].concrete_shape
            return (-1, -1, -1, -1)

    N, IC, IH, IW = get_shape(data)
    OC, IC, KH, KW = get_shape(weight)
    N, OC, OH, OW = get_shape(out_type)
    PH_L, PW_L, PH_R, PW_R = attrs.get_int_tuple("padding")
    SH, SW = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    G = int(attrs.groups)
    new_attrs = dict(attrs)

    if G>1: # for mobilenet
        IC = IC * G
        new_attrs['data_layout'] = "NCHW"
        new_attrs['kernel_layout'] = "OIHW"
        new_attrs['out_layout'] = "NCHW"
        return relay.nn.conv2d(data, weight, **new_attrs)

    res = relay.query_layout.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G)

    src_df, weight_df, dst_df = res.split(',')

    def trans_data(input_data, is_weight=False):
        dic = data_dic
        res = input_data
        if is_weight:
            dic = weight_dic
                
        for key, value in dic.items():
            if key.upper() in input_data:
                res = res.replace(key.upper(), value, 1)
                res = res.replace(key, value.lower(), 1)
            else:
                res = res.replace(key, value, 1)
        return res

    new_attrs['data_layout'] = trans_data(src_df, is_weight=False)
    new_attrs['kernel_layout'] = trans_data(weight_df, is_weight=True)
    new_attrs['out_layout'] = trans_data(dst_df, is_weight=False)

    return relay.nn.conv2d(data, weight, **new_attrs)

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    if network=="InceptionV3":
        input_shape = (batch_size, 3, 300, 300)
    if network=="i3d_resnet50_v1_kinetics400":
        input_shape = (batch_size, 3, 20, 224, 224)
    
    block = gluoncv.model_zoo.get_model(network_dict[network], pretrained=True)
    mod, params = relay.frontend.from_mxnet(
        block, shape={"data": input_shape}, dtype=dtype
    )

    seq = tvm.transform.Sequential(
        [   
            # tvm.transform.PrintIR(),
            relay.transform.CanonicalizeOps(),
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            # tvm.transform.PrintIR(),

            relay.transform.SimplifyExpr(),
            relay.transform.FoldConstant(),
            # tvm.transform.PrintIR(),
            
            relay.transform.AlterOpLayout(),
            relay.transform.FoldConstant(),
            # tvm.transform.PrintIR(),

            relay.transform.MergeComposite(pattern_table()),
            # tvm.transform.PrintIR(),
            relay.transform.AnnotateTarget("dnnl"),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
            # tvm.transform.PrintIR(),
        ]
    )


    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):# 
        json, lib, params = relay.build(seq(mod), target=target, params=params)

    if check_acc:
        img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        img_name = "cat.png"
        img_path = download_testdata(img_url, img_name, module="data")
        image = Image.open(img_path).resize((input_shape[2], input_shape[3]))
        sample = transform_image(image)
        if batch_size>1:
            sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])
        if network=="i3d_resnet50_v1_kinetics400":
            sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3], input_shape[4])

        import tvm.contrib.graph_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
    
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        out = rt_mod.run()
        sample_for_mxnet = mx.ndarray.array(sample)
        mxnet_output = block(sample_for_mxnet)
        tvm_output = rt_mod.get_output(0)
        # print("mxnet_output:{}".format(mxnet_output))
        # print("tvm_output:{}".format(tvm_output))
        print("{} mse:{}".format(network, np.mean((tvm_output.asnumpy()-mxnet_output.asnumpy())**2)))
    elif profiling:
        import datetime
        tic = datetime.datetime.now()
        from tvm.contrib.debugger import debug_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
        sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])#np.ones((batch_size, 3, 224, 224))#
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        total_time_lst = []
        for i in range(batches+warmup):
            tmp = rt_mod.profile()
            gap = tmp.calls[1]["Duration (us)"].microseconds
            #percent = tmp.calls[0]["Percent"].percent
            reorder = tmp.calls[2]["Duration (us)"].microseconds
            #total_time = us * 100 / percent / 1000
            print("{}/{}: gap:{:.4f}, reorder:{:.4f}".format(i, batches+warmup, gap, reorder))
            total_time = gap+reorder
            total_time_lst.append(total_time)
        print("network:{}".format(network))
        print("all ops' execution time:{}".format(np.mean(total_time_lst[warmup::])))
        print("all ops' execution time:{}".format(np.mean(total_time_lst[warmup::])/1000))
        print("profiling time:{}".format(datetime.datetime.now()-tic))
    
    else:
        import tvm.contrib.graph_executor as graph_executor
        rt_mod = graph_executor.create(json, lib, ctx)
        sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = rt_mod.run()
        with_fuse_fps = batches * batch_size / (time.time() - tic)
        print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))
        
if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "vgg11", "vgg13", "vgg16", "vgg19", 
                "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                "densenet121", "InceptionV3", "MobileNet1.0", "i3d_resnet50_v1_kinetics400", "all"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=cascadelake -model=platinum-8280",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--check_acc", type=bool, default=False)
    args = parser.parse_args()

    if args.network == "all":
        networks = [
                    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                    "vgg11", "vgg13", "vgg16", "vgg19", 
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                    "densenet121", 
                    "InceptionV3",
                    ]
    else:
        networks = [args.network]

    target = tvm.target.Target(args.target)

    for network in networks:
        benchmark(network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
        warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
