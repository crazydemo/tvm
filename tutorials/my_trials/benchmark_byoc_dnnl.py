'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
from typing import Tuple
import gluoncv
from mxnet.gluon.model_zoo import vision
from gluoncv.utils import export_block
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
import onnx
import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl

import numpy as np

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def partition_for_dnnl(mod, params=None, alter_layout=True):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with TempOpAttr("nn.conv2d", "FTVMLegalize", dnnl.legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", dnnl.legalize_group_conv):
            seq = tvm.transform.Sequential(
                [
                    transform.CanonicalizeOps(),
                    transform.InferType(),
                    transform.SimplifyInference(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    # tvm.transform.PrintIR(),
                    transform.SimplifyExpr(),
                    transform.FoldConstant(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    if alter_layout:
        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", dnnl.alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", dnnl.alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", dnnl.alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                        ):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    transform.AlterOpLayout(),
                                    transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            # tvm.transform.PrintIR(),
            transform.MergeComposite(dnnl.pattern_table()),
            transform.AnnotateTarget("dnnl"),
            # tvm.transform.PrintIR(),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        print(mod)
        mod = dnnl.prune_dnnl_subgraphs(mod)
        print("======================================")
        print(mod)
    return mod

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    if network=="InceptionV3":
        input_shape = (batch_size, 3, 300, 300)
    if network=="i3d_resnet50_v1_kinetics400":
        input_shape = (batch_size, 3, 20, 224, 224)
    
    if network=="3DUnet":
        input_shape = (batch_size, 4, 160, 224, 224)

        onnx_model = onnx.load("/home2/zhangya9/onnx_models/3DUnet_224_224_160.onnx")
        mod, params = relay.frontend.from_onnx(
            onnx_model, shape={"input": input_shape}, dtype=dtype
        )
    else:
        block = gluoncv.model_zoo.get_model(network, pretrained=True)
        # print(type(block))
        # export_block('resnet18_v1', block, preprocess=False, layout='CHW')
        # sys = "/home2/zhangya9/tvm/build/resnet18_v1-symbol.json"
        # params = "/home2/zhangya9/tvm/build/resnet18_v1-0000.params"
        # onnx_file = "/home2/zhangya9/tvm/build/resnet18_v1_gluoncv.onnx"
        # converted_model_path = onnx_mxnet.export_model(sys, params, [(1, 3, 224, 224)], np.float32, onnx_file)
        # # block = vision.resnet18_v1(pretrained=True, ctx=mx.cpu())
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        # onnx_model = onnx.load("/home2/zhangya9/tvm/build/resnet18_v1_gluoncv.onnx")
        # mod, params = relay.frontend.from_onnx(
        #     onnx_model, shape={"data": (1, 3, 224, 224)}, dtype=dtype
        # )
        # print(mod, params)
        # print(mod.astext(show_meta_data=False))
    sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])
    if len(input_shape)>4:
        sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3], input_shape[4])

    processed_mod = partition_for_dnnl(mod, params, alter_layout=True)
    print(processed_mod)    
    with tvm.transform.PassContext(opt_level=3):
        json, lib, params = relay.build(processed_mod, target=target, params=params)
    # print(json)
    import tvm.contrib.graph_executor as graph_executor
    # from tvm.contrib.debugger import debug_executor as graph_executor
    rt_mod = graph_executor.create(json, lib, ctx)
    
    rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
    rt_mod.set_input(**params)
    # out = rt_mod.run()
    
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
        # print("================start run=========================")
        rt_mod.run()
        # print(rt_mod.profile())
    with_fuse_fps = batches * batch_size / (time.time() - tic)
    print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))


def benchmark_onnx_model(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    import onnx
    import os
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    if "inception" in network:
        shape_dic={"data_0": input_shape}
    elif "mobilenet" in network:
        shape_dic={"input": input_shape}
    else:
        shape_dic={"data": input_shape}

    root = "/home2/zhangya9/onnx_models"
    onnx_model_dic = {"3DUnet": "3DUnet_224_224_160.onnx",
                      "resnet18": "resnet18-v1-7.onnx",
                      "resnet34": "resnet34-v1-7.onnx",
                      "resnet50": "resnet50-v1-7.onnx",
                      "resnet101": "resnet101-v1-7.onnx",
                      "resnet152": "resnet152-v1-7.onnx",
                      "vgg16": "vgg16-7.onnx",
                      "vgg16_bn": "vgg16-bn-7.onnx",
                      "vgg19": "vgg19-7.onnx",
                      "vgg19_bn": "vgg19-bn-7.onnx",
                      "inception_v1": "inception-v1-12.onnx",
                      "inception_v2": "inception-v2-9.onnx",
                      "mobilenet_v2": "mobilenetv2-7.onnx",}
    
    path = os.path.join(root, onnx_model_dic[network])
    onnx_model = onnx.load(path)
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=shape_dic, dtype=dtype
    )
    # print(mod, params)
    sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])

    processed_mod = dnnl.partition_for_dnnl(mod, params, alter_layout=True)
    with tvm.transform.PassContext(opt_level=3):
        json, lib, params = relay.build(processed_mod, target=target, params=params)
    import tvm.contrib.graph_executor as graph_executor
    # from tvm.contrib.debugger import debug_executor as graph_executor
    rt_mod = graph_executor.create(json, lib, ctx)
    
    rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
    rt_mod.set_input(**params)
    
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
        # print("================start run=========================")
        rt_mod.run()
        # print(rt_mod.profile())
    with_fuse_fps = batches * batch_size / (time.time() - tic)
    print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))

if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["ResNet18_v1", "ResNet34_v1", "ResNet50_v1", "ResNet101_v1", "ResNet152_v1",
                "vgg11", "vgg13", "vgg16", "vgg19", 
                "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                "densenet121", "InceptionV3", "MobileNet1.0", "ResNext50_32x4d", "i3d_resnet50_v1_kinetics400", "3DUnet", "all", "onnx"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size")
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
                    "ResNet18_v2", 
                    # "ResNet34_v1", 
                    # "ResNet50_v1", 
                    # "ResNet101_v1", 
                    # "ResNet152_v1",
                    # "VGG16", 
                    # "VGG16_bn", 
                    # "VGG19", 
                    # "VGG19_bn", 
                    # "MobileNetV2_1.0",
                    ]
    elif args.network == "onnx":
        networks = [
                    # "resnet18",
                    # "resnet34",
                    # "resnet50", 
                    # "resnet101",
                    #  "resnet152",
                    # "vgg16", 
                    # "vgg16_bn", 
                    # "vgg19", 
                    # "vgg19_bn",
                    # "inception_v1",
                    # "inception_v2",
                    # "mobilenet_v2",
                    "3DUnet"
                    ]
    else:
        networks = [args.network]

    target = tvm.target.Target(args.target)

    if args.network != "onnx":
        for network in networks:
            benchmark(network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
            warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
    else:
        for network in networks:
            benchmark_onnx_model(network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
            warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
