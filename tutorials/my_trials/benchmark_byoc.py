'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
from tkinter.tix import Tree
from typing import Tuple
import os
import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
from onnxruntime.backend.backend import OnnxRuntimeBackend as backend

import numpy as np
import time

network_root_path = "/home2/zhangya9/onnx_models"
network_dic = { "MobileNet-v2-1.0": "MobileNet/torch_model/mobilenetv2_torch.onnx",
                "resnet50-v1": "ResNet/v1/resnet50v1/resnet50-v1-7.onnx",
                "resnet50-v2": "ResNet/v2/resnet50v2/resnet50-v2-7.onnx",
                "squeezenet1.0": "SqueezeNet/squeezenet/model.onnx",
                "squeezenet1.1": "SqueezeNet/squeezenet1.1/squeezenet1.1.onnx",
                "vgg16": "VGG/vgg16/vgg16.onnx",
                "vgg16-bn": "VGG/vgg16-bn/vgg16-bn.onnx",
                "densenet121": "DenseNet-121/densenet121/model.onnx",
                "inception_v3": "Inception_V3/torchvision_model/inception_v3.onnx",
                "shufflenet_v2": "ShuffleNet_V2/torch_model/shufflenet_v2_x1_0.onnx",
                "efficientnet-b0-pytorch": "efficientnet-b0-pytorch/efficientnet-b0.onnx",
                "resnext50_32x4d": "ResNext/torch_model/resnext50_32x4d.onnx",
                "wide_resnet50_2": "Wide_ResNet/torch_model/wide_resnet50_2.onnx",
                "resnest50": "ResNeSt/torch_model/resnest50.onnx",
              }

input_dic = { "MobileNet-v2-1.0": "input",
              "resnet50-v1": "data",
              "resnet50-v2": "data",
              "squeezenet1.0": "data_0",
              "squeezenet1.1": "data",
              "vgg16": "data",
              "vgg16-bn": "data",
              "densenet121": "data_0",
              "inception_v3": "input",
              "shufflenet_v2": "input",
              "efficientnet-b0-pytorch": "data",
              "resnext50_32x4d": "input",
              "wide_resnet50_2": "input",
              "resnest50": "input",
            }

shape_dic = { "MobileNet-v2-1.0": [1, 3, 224, 224],
              "resnet50-v1": [1, 3, 224, 224],
              "resnet50-v2": [1, 3, 224, 224],
              "squeezenet1.0": [1, 3, 224, 224],
              "squeezenet1.1": [1, 3, 224, 224],
              "vgg16": [1, 3, 224, 224],
              "vgg16-bn": [1, 3, 224, 224],
              "densenet121": [1, 3, 224, 224],
              "inception_v3": [1, 3, 299, 299],
              "shufflenet_v2": [1, 3, 224, 224],
              "efficientnet-b0-pytorch": [1, 3, 224, 224],
              "resnext50_32x4d": [1, 3, 224, 224],
              "wide_resnet50_2": [1, 3, 224, 224],
              "resnest50": [1, 3, 224, 224],
            }

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
                    # tvm.transform.PrintIR(),
                    transform.CanonicalizeOps(),
                    transform.InferType(),
                    transform.SimplifyInference(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    # tvm.transform.PrintIR(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    transform.SimplifyExpr(),
                    transform.FoldConstant(),
                    # tvm.transform.PrintIR(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                    # tvm.transform.PrintIR(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)

    mod = dnnl.rewrite_resnetv1(mod)

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
                                    # tvm.transform.PrintIR(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(dnnl.pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        mod = dnnl.prune_dnnl_subgraphs(mod)
    return mod

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()
    input_name = input_dic[network]
    input_shape = shape_dic[network]
    input_shape[0] = batch_size
    # print(input_shape)
    shape_dict = {input_name: input_shape}
    model_path = os.path.join(network_root_path, network_dic[network])
    sample = np.random.randint(0, 1, input_shape).astype(dtype)
    if network_dic[network] == "":
        print("=============converting torch model===============")
        import torch
        import torchvision

        model_name = network
        # get list of models
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        # load pretrained models, using ResNeSt-50 as an example
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        model.eval()
        # model = getattr(torchvision.models, model_name)(pretrained=True)
        # model = model.eval()

        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif model_path.split(".")[-1] == "pb":
        print("=============converting TF model===============")
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        import tvm.relay.testing.tf as tf_testing
        layout = layout_dic[network]
        with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
    elif model_path.split(".")[-1] == "onnx":
        print("=============converting ONNX model===============")
        import onnx
        onnx_model = onnx.load(model_path)
        # print(onnx_model.graph.input)
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        # print(mod)
    elif model_path.split(".")[-1] == "caffemodel":
        print("=============converting caffe model===============")
        import caffe.proto.caffe_pb2 as caffe_pb2
        model = caffe_pb2.NetParameter()
        f = open(model_path, 'rb')
        model.ParseFromString(f.read())
        mod, params = relay.frontend.from_caffe2(
                        model, shape_dict)
    elif model_path.split(".")[-1] == "json":
        print("=============converting MXNet model===============")
        import mxnet as mx
        json_file = model_path
        params_file = model_path.replace("symbol.json", "0000.params")
        mod = mx.gluon.nn.SymbolBlock(outputs=mx.sym.load(json_file), inputs=mx.sym.var('data'))
        mod.load_params(params_file, ctx=ctx)
        mod, params = relay.frontend.from_mxnet(mod, shape_dict)
    else:
        print("Unsupported model type!")

    print("=============Optimizing===============")
    print(mod)
    processed_mod = partition_for_dnnl(mod, params, alter_layout=True)
    print(processed_mod)

    print("=============Building===============")
    with tvm.transform.PassContext(opt_level=3):
        json, lib, params = relay.build(processed_mod, target=target, params=params)
    # print(json)
    import tvm.contrib.graph_executor as graph_executor
    # from tvm.contrib.debugger import debug_executor as graph_executor
    rt_mod = graph_executor.create(json, lib, ctx)
    
    rt_mod.set_input(input_name, tvm.nd.array(sample.astype(dtype)))
    rt_mod.set_input(**params)

    print("=============Checking accuracy===============")
    if check_acc and batch_size==1:
        onnx_output = list(backend.run_model(onnx_model, sample))
        rt_mod.run()
        tvm_output = rt_mod.get_output(0)
        print(network, np.testing.assert_almost_equal(onnx_output, [tvm_output.asnumpy()], 4))
    # out = rt_mod.run()
    print("=============Running===============")
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
        # starttime = time.time()
        rt_mod.run()
        # endtime = time.time()
        # print("exe time this run: %.8s s" % str(endtime-starttime))
        # print(rt_mod.profile())
    with_fuse_fps = batches * batch_size / (time.time() - tic)
    print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))

if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="resnet50-v1",
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
    
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--check_acc", type=bool, default=True)
    args = parser.parse_args()
    target = tvm.target.Target(args.target)

    if args.network == "all":
        for net in [
                    "MobileNet-v2-1.0",
                    "resnet50-v1",
                    "resnet50-v2",
                    "squeezenet1.0",
                    "squeezenet1.1",
                    "vgg16",
                    "vgg16-bn",
                    "densenet121",
                    "inception_v3",
                    "shufflenet_v2",
                    "efficientnet-b0-pytorch",
                    "resnext50_32x4d",
                    "wide_resnet50_2",
                    "resnest50"]:
            print("################checking {}#################".format(net))
            benchmark(net, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
            warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
    else:
        benchmark(args.network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
            warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)